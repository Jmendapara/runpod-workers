"""RunPod serverless handler: train one LoRA with ostris/ai-toolkit.

Job lifecycle (every phase is reported through runpod.serverless.progress_update, which the
OpenSourceGen server reads from /status while the job is IN_PROGRESS):
  preparing  — download the dataset from R2, write captions, write the ai-toolkit config
  loading    — ai-toolkit loads / downloads the base model (HF cache on the network volume)
  caching    — latent / text-embedding caching
  training   — step N/total, sec/step, ETA, loss (parsed from ai-toolkit's tqdm line)
  sampling   — preview image(s) at the final step (image models only)
  uploading  — the .safetensors + samples go to R2
Cancellation: the server exposes a per-training control URL; the handler polls it every
CONTROL_POLL_S seconds and kills the trainer when `cancel` is true (RunPod's /cancel does not
reliably stop a running handler). The result then carries `cancelled: true`.
"""
from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import requests
import runpod
import yaml

from r2 import download_to_file, make_client, resolve_bucket_target, upload_file
from run_log import RunLog
from trainer_config import STEP_RE, build_config, caption_for, parse_progress_line, pick_output_loras, previews_ready, sample_step_of, validate_input

AI_TOOLKIT_DIR = os.environ.get("AI_TOOLKIT_DIR", "/app/ai-toolkit")
JOBS_ROOT = os.environ.get("TRAINER_JOBS_ROOT", "/workspace/jobs")
PROGRESS_EVERY_S = float(os.environ.get("PROGRESS_EVERY_S", "5"))
CONTROL_POLL_S = float(os.environ.get("CONTROL_POLL_S", "20"))
DOWNLOAD_THREADS = int(os.environ.get("DOWNLOAD_THREADS", "8"))
LOG_TAIL_LINES = 80
MAX_SAMPLES = 4
PREVIEW_SCAN_S = 5.0
PREVIEW_SETTLE_S = 2.0  # a sample file untouched for this long has finished being written
REFRESH_WORKER = os.environ.get("REFRESH_WORKER", "false").lower() == "true"


STEP_LOG_EVERY = 100
RUN_LOG_FILE = "logs/run.log"
WORKER_ID = os.environ.get("RUNPOD_POD_ID")

run_log: RunLog | None = None  # the current job's log; a worker runs one job at a time


def log(msg: str) -> None:
    print(f"lora-trainer - {msg}", flush=True)
    if run_log:
        run_log.write(f"lora-trainer - {msg}")


def hf_home() -> str:
    """Network volume when attached (persists the model cache across workers), else container disk."""
    explicit = os.environ.get("HF_HOME")
    if explicit:
        return explicit
    if os.path.isdir("/runpod-volume"):
        return "/runpod-volume/hf"
    return "/workspace/hf"


def gpu_name() -> str | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=gpu_name,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        line = (out.stdout or "").strip().splitlines()
        return line[0].strip() if line else None
    except Exception:
        return None


class Progress:
    """Throttled progress_update() publisher shared by the reader thread and the main loop."""

    def __init__(self, job, total_steps: int):
        self.job = job
        self.total = total_steps
        self.lock = threading.Lock()
        self.phase = "preparing"
        self.step = 0
        self.loss: float | None = None
        self.remaining_s: int | None = None
        self.message: str | None = None
        self.started = time.time()
        self.gpu = gpu_name()
        self.train_started: float | None = None
        self.step_times: deque[tuple[float, int]] = deque(maxlen=50)
        self.last_push = 0.0
        self.last_payload: dict | None = None
        self.previews: list[dict] = []

    def add_preview(self, step: int, key: str) -> None:
        with self.lock:
            entry = next((p for p in self.previews if p["step"] == step), None)
            if entry is None:
                entry = {"step": step, "keys": []}
                self.previews.append(entry)
                self.previews.sort(key=lambda p: p["step"])
            entry["keys"].append(key)

    def set_phase(self, phase: str, message: str | None = None) -> None:
        with self.lock:
            if phase == "training" and self.train_started is None:
                self.train_started = time.time()
            self.phase = phase
            self.message = message

    def set_step(self, step: int, loss: float | None, remaining_s: int | None) -> None:
        with self.lock:
            if self.train_started is None:
                self.train_started = time.time()
            self.phase = "training"
            if step > self.step:
                self.step_times.append((time.time(), step))
            self.step = max(self.step, step)
            if loss is not None:
                self.loss = loss
            if remaining_s is not None:
                self.remaining_s = remaining_s

    def sec_per_step(self) -> float | None:
        if len(self.step_times) < 2:
            return None
        (t0, s0), (t1, s1) = self.step_times[0], self.step_times[-1]
        if s1 <= s0 or t1 <= t0:
            return None
        return (t1 - t0) / (s1 - s0)

    def payload(self) -> dict:
        with self.lock:
            sps = self.sec_per_step()
            eta = self.remaining_s
            if eta is None and sps and self.total:
                eta = int(max(0, self.total - self.step) * sps)
            return {
                "phase": self.phase,
                "step": self.step,
                "total_steps": self.total,
                "pct": round(100.0 * self.step / self.total, 1) if self.total else 0,
                "elapsed_s": int(time.time() - self.started),
                "train_elapsed_s": int(time.time() - self.train_started) if self.train_started else 0,
                "eta_s": eta,
                "sec_per_step": round(sps, 3) if sps else None,
                "loss": self.loss,
                "message": self.message,
                "previews": [{"step": p["step"], "keys": list(p["keys"])} for p in self.previews],
                "worker_id": WORKER_ID,
                "gpu": self.gpu,
                "updated_at": int(time.time()),
            }

    def push(self, force: bool = False) -> None:
        now = time.time()
        if not force and now - self.last_push < PROGRESS_EVERY_S:
            return
        payload = self.payload()
        if not force and payload == self.last_payload:
            return
        self.last_push = now
        self.last_payload = payload
        try:
            runpod.serverless.progress_update(self.job, payload)
        except Exception as exc:  # never let telemetry kill a training
            log(f"progress_update failed: {exc}")


def download_dataset(client, bucket: str, job: dict, workdir: str) -> dict:
    ds_dir = os.path.join(workdir, "dataset")
    os.makedirs(ds_dir, exist_ok=True)
    items = job["dataset"]["items"]
    trigger = job.get("trigger_word")
    default_caption = job["dataset"].get("default_caption")

    def fetch(idx_item):
        idx, item = idx_item
        stem = f"item_{idx:04d}"
        path = os.path.join(ds_dir, stem + item["ext"])
        size = download_to_file(client, bucket, item["r2_key"], path)
        with open(os.path.join(ds_dir, stem + ".txt"), "w", encoding="utf-8") as fh:
            fh.write(caption_for(item, trigger, default_caption))
        return size

    with ThreadPoolExecutor(max_workers=DOWNLOAD_THREADS) as pool:
        sizes = list(pool.map(fetch, enumerate(items)))
    return {
        "count": len(items),
        "images": sum(1 for it in items if it["kind"] == "image"),
        "videos": sum(1 for it in items if it["kind"] == "video"),
        "bytes": sum(sizes),
    }


class Trainer:
    """Runs `python run.py config.yaml` and parses its output."""

    def __init__(self, config_path: str, env: dict, progress: Progress):
        self.config_path = config_path
        self.env = env
        self.progress = progress
        self.tail: deque[str] = deque(maxlen=LOG_TAIL_LINES)
        self.next_logged_step = 0
        self.proc: subprocess.Popen | None = None
        self.reader: threading.Thread | None = None

    def start(self) -> None:
        self.proc = subprocess.Popen(
            [sys.executable, "run.py", self.config_path],
            cwd=AI_TOOLKIT_DIR,
            env=self.env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=0,
            start_new_session=True,  # own process group → terminate() reaches dataloader workers
        )
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.reader.start()

    def _read(self) -> None:
        assert self.proc and self.proc.stdout
        buf = b""
        while True:
            chunk = self.proc.stdout.read(4096)
            if not chunk:
                break
            buf += chunk
            # tqdm redraws with \r; treat \r and \n both as line ends
            while True:
                cut = -1
                for sep in (b"\n", b"\r"):
                    i = buf.find(sep)
                    if i != -1 and (cut == -1 or i < cut):
                        cut = i
                if cut == -1:
                    break
                line = buf[:cut].decode("utf-8", "replace")
                buf = buf[cut + 1:]
                self._on_line(line)
        if buf:
            self._on_line(buf.decode("utf-8", "replace"))

    def _on_line(self, line: str) -> None:
        text = line.strip()
        if not text:
            return
        if not is_step_redraw(text):
            self.tail.append(text[:400])
            print(f"[ai-toolkit] {text[:400]}", flush=True)
            if run_log:
                run_log.write(f"[ai-toolkit] {text[:400]}")
        parsed = parse_progress_line(text, self.progress.total)
        if not parsed:
            return
        if "step" in parsed:
            self.progress.set_step(parsed["step"], parsed.get("loss"), parsed.get("remaining_s"))
            if parsed["step"] >= self.next_logged_step:
                self.next_logged_step = parsed["step"] + STEP_LOG_EVERY
                log(f"step {parsed['step']}/{self.progress.total} loss={self.progress.loss} sec_per_step={self.progress.sec_per_step()}")
            return
        phase = parsed.get("phase")
        if phase == "sampling":
            self.progress.set_phase("sampling", text[:160])
        elif phase == "saving":
            self.progress.set_phase("uploading", text[:160])
        elif phase in ("loading", "caching") and self.progress.phase in ("preparing", "loading", "caching"):
            self.progress.set_phase(phase, text[:160])

    def poll(self) -> int | None:
        return self.proc.poll() if self.proc else None

    def terminate(self, grace_s: float = 25.0) -> None:
        if not self.proc or self.proc.poll() is not None:
            return
        try:
            os.killpg(self.proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        deadline = time.time() + grace_s
        while time.time() < deadline and self.proc.poll() is None:
            time.sleep(0.5)
        if self.proc.poll() is None:
            try:
                os.killpg(self.proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

    def wait_reader(self) -> None:
        if self.reader:
            self.reader.join(timeout=10)


def is_step_redraw(text: str) -> bool:
    """True for tqdm step redraws (kept out of the log tail — hundreds per minute)."""
    return bool(STEP_RE.search(text)) and ("it/s" in text or "s/it" in text)


class ControlPoller:
    """Asks the server whether the user cancelled. Fails open (keeps training) on errors."""

    def __init__(self, url: str | None):
        self.url = url
        self.last = 0.0
        self.cancel = False

    def check(self) -> bool:
        if not self.url or self.cancel:
            return self.cancel
        now = time.time()
        if now - self.last < CONTROL_POLL_S:
            return False
        self.last = now
        try:
            r = requests.get(self.url, timeout=8)
            if r.ok:
                data = r.json()
                if isinstance(data, dict) and data.get("cancel") is True:
                    self.cancel = True
                    log("cancel requested by the server")
        except Exception as exc:
            log(f"control poll failed (ignored): {exc}")
        return self.cancel


def find_output_loras(output_dir: str, name: str) -> list[tuple[str | None, str]]:
    """(stage, path) for every file the run produced. See trainer_config.pick_output_loras."""
    folder = os.path.join(output_dir, name)
    if not os.path.isdir(folder):
        return []
    return [(stage, os.path.join(folder, file)) for stage, file in pick_output_loras(os.listdir(folder), name)]


def lora_problem(path: str) -> str | None:
    """Why this LoRA would do nothing or break an image: every output weight zero (ai-toolkit #1046,
    #1054) or a non-finite weight (#975). A run like that looks fine until someone uses the file."""
    from safetensors import safe_open

    outputs = 0
    alive = 0
    with safe_open(path, framework="pt", device="cpu") as weights:
        for key in weights.keys():
            tensor = weights.get_tensor(key).float()
            if not tensor.isfinite().all():
                return f"non-finite weights in {key}"
            if "lora_B" in key or "lora_up" in key:
                outputs += 1
                if tensor.abs().max() > 0:
                    alive += 1
    if outputs == 0:
        return "no LoRA weights in the file"
    if alive == 0:
        return f"all {outputs} output layers are zero"
    return None


class PreviewUploader:
    """Uploads the samples ai-toolkit renders on the way, as they land on disk, so the app can show
    them while it trains. A failed upload is logged and retried; it never touches the training."""

    def __init__(self, client, bucket: str, prefix: str, folder: str, progress: Progress):
        self.client = client
        self.bucket = bucket
        self.prefix = prefix
        self.folder = folder
        self.progress = progress
        self.done: set[str] = set()
        self.counts: dict[int, int] = {}
        self.last_scan = 0.0

    def scan(self, force: bool = False) -> None:
        now = time.time()
        if not force and now - self.last_scan < PREVIEW_SCAN_S:
            return
        self.last_scan = now
        try:
            names = os.listdir(self.folder)
        except OSError:
            return
        uploaded = False
        for step, name in previews_ready(names, self.progress.total, self.done):
            path = os.path.join(self.folder, name)
            try:
                if not force and now - os.path.getmtime(path) < PREVIEW_SETTLE_S:
                    continue
                index = self.counts.get(step, 0)
                key = f"{self.prefix}samples/step-{step:06d}-{index}{os.path.splitext(name)[1].lower()}"
                upload_file(self.client, self.bucket, key, path)
            except Exception as exc:
                log(f"preview upload failed ({name}): {exc}")
                continue
            self.done.add(name)
            self.counts[step] = index + 1
            self.progress.add_preview(step, key)
            uploaded = True
        if uploaded:
            self.progress.push(force=True)


def find_samples(output_dir: str, name: str) -> list[str]:
    folder = os.path.join(output_dir, name, "samples")
    if not os.path.isdir(folder):
        return []
    files = [f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))]
    if not files:
        return []
    # filenames: <time>__<step:09d>_<i>.<ext>; keep the highest step only
    best = max(sample_step_of(f) for f in files)
    picked = sorted(f for f in files if sample_step_of(f) == best)[:MAX_SAMPLES]
    return [os.path.join(folder, f) for f in picked]


def handler(job):
    """Runs one training and adds what an investigation needs to every result, failed or not."""
    global run_log
    run = {"started": time.time(), "progress": None}
    result = train(job, run)
    progress = run["progress"]
    info = {"worker_id": WORKER_ID, "gpu": gpu_name(), "total_seconds": int(time.time() - run["started"])}
    if progress:
        info.update({"phase": progress.phase, "steps_completed": progress.step})
    if run_log:
        info["log_key"] = run_log.key
    result = {**info, **result}
    if result.get("error"):
        log(f"failed: code={result.get('code')} phase={info.get('phase')} step={info.get('steps_completed')} error={str(result['error'])[:1000]}")
    if run_log:
        run_log.upload(force=True)
        run_log.close()
        run_log = None
    return result


def train(job, run):
    global run_log
    job_input = job.get("input") or {}
    job_id = job.get("id", "?")
    started = run["started"]
    workdir = None
    trainer: Trainer | None = None
    try:
        try:
            spec = validate_input(job_input)
        except ValueError as exc:
            return {"error": str(exc), "code": "invalid_input"}

        try:
            target = resolve_bucket_target(spec["bucket_profile"])
        except (ValueError, RuntimeError) as exc:
            return {"error": str(exc), "code": "storage_not_configured"}

        progress = Progress(job, spec["train"]["steps"])
        run["progress"] = progress
        progress.push(force=True)

        workdir = os.path.join(JOBS_ROOT, spec["training_id"])
        shutil.rmtree(workdir, ignore_errors=True)
        os.makedirs(workdir, exist_ok=True)

        client = make_client(target)
        log_key = spec["output"]["prefix"] + RUN_LOG_FILE
        run_log = RunLog(
            os.path.join(JOBS_ROOT, f"{spec['training_id']}.log"), log_key,
            lambda path: upload_file(client, target.bucket, log_key, path, "text/plain"),
            secrets=[spec["hf_token"], spec["control_url"]],
        )
        log(f"job={job_id} training={spec['training_id']} worker={WORKER_ID} gpu={progress.gpu} "
            f"arch={spec['model']['arch']} items={len(spec['dataset']['items'])} steps={spec['train']['steps']} "
            f"bucket={target.bucket} input_bucket={target.input_bucket}")
        run_log.upload(force=True)
        try:
            ds_stats = download_dataset(client, target.input_bucket, spec, workdir)
        except Exception as exc:
            traceback.print_exc()
            return {"error": f"Failed to download the training dataset: {exc}", "code": "dataset_download_failed"}
        log(f"dataset ready: {ds_stats}")

        config = build_config(spec, workdir)
        config_path = os.path.join(workdir, "config.yaml")
        with open(config_path, "w", encoding="utf-8") as fh:
            yaml.safe_dump(config, fh, sort_keys=False)
        safe_cfg = json.loads(json.dumps(config))
        log(f"config: {json.dumps(safe_cfg)[:8000]}")

        env = dict(os.environ)
        env["HF_HOME"] = hf_home()
        os.makedirs(env["HF_HOME"], exist_ok=True)
        env["PYTHONUNBUFFERED"] = "1"
        env.setdefault("TQDM_MININTERVAL", "2")
        if spec["hf_token"]:
            env["HF_TOKEN"] = spec["hf_token"]
            env["HUGGING_FACE_HUB_TOKEN"] = spec["hf_token"]
        env.pop("hf_token", None)

        progress.set_phase("loading", "Loading the base model")
        progress.push(force=True)
        trainer = Trainer(config_path, env, progress)
        trainer.start()
        control = ControlPoller(spec["control_url"])
        previews = PreviewUploader(
            client, target.bucket, spec["output"]["prefix"],
            os.path.join(workdir, "output", spec["name"], "samples"), progress,
        )

        cancelled = False
        while True:
            rc = trainer.poll()
            if rc is not None:
                break
            if control.check():
                cancelled = True
                trainer.terminate()
                break
            previews.scan()
            progress.push()
            run_log.upload()
            time.sleep(1.0)
        trainer.wait_reader()
        previews.scan(force=True)
        rc = trainer.poll()
        tail = list(trainer.tail)

        if cancelled:
            log(f"cancelled at step {progress.step}/{progress.total}")
            return {
                "cancelled": True,
                "steps_completed": progress.step,
                "train_seconds": int(time.time() - (progress.train_started or started)),
                "log_tail": tail[-20:],
            }
        output_dir = os.path.join(workdir, "output")
        lora_paths = find_output_loras(output_dir, spec["name"])
        # A crash after the final save (usually the last preview) must not throw away a finished LoRA.
        if rc != 0 and lora_paths and progress.step >= progress.total:
            log(f"ai-toolkit exited with {rc} after the last step; keeping the saved LoRA")
        elif rc != 0:
            log(f"ai-toolkit exited with {rc}")
            err_line = next((l for l in reversed(tail) if "Error" in l or "error" in l or "Exception" in l), None)
            code = "oom" if any("out of memory" in l.lower() for l in tail) else "training_failed"
            return {
                "error": (err_line or f"Training process exited with code {rc}")[:500],
                "code": code,
                "steps_completed": progress.step,
                "log_tail": tail,
            }

        if not lora_paths:
            return {"error": "Training finished but no LoRA file was produced.", "code": "no_output", "log_tail": tail}
        for _, lora_path in lora_paths:
            problem = lora_problem(lora_path)
            if problem:
                log(f"rejecting {os.path.basename(lora_path)}: {problem}")
                return {"error": "This training produced a LoRA that would not work, so your credits were refunded. Trying different settings may help.", "code": "bad_lora", "log_tail": [problem, *tail[-20:]]}

        progress.set_phase("uploading", "Uploading your LoRA")
        progress.push(force=True)
        prefix = spec["output"]["prefix"]
        stem = os.path.splitext(spec["output"]["filename"])[0]
        lora_files = []
        for stage, lora_path in lora_paths:
            key = f"{prefix}{stem}_{stage}_noise.safetensors" if stage else prefix + spec["output"]["filename"]
            lora_files.append({"stage": stage, "key": key, "bytes": upload_file(client, target.bucket, key, lora_path)})
        lora_key = lora_files[-1]["key"]
        lora_bytes = sum(f["bytes"] for f in lora_files)
        sample_keys = []
        for i, sp in enumerate(find_samples(output_dir, spec["name"])):
            key = f"{prefix}samples/{i}{os.path.splitext(sp)[1].lower()}"
            try:
                upload_file(client, target.bucket, key, sp)
                sample_keys.append(key)
            except Exception as exc:
                log(f"sample upload failed ({key}): {exc}")
        # the config is useful for support / reproducibility, and tiny
        try:
            upload_file(client, target.bucket, prefix + "config.yaml", config_path, "text/yaml")
        except Exception as exc:
            log(f"config upload failed: {exc}")

        train_seconds = int(time.time() - (progress.train_started or started))
        result = {
            "lora_key": lora_key,
            "lora_bytes": lora_bytes,
            "lora_files": [f for f in lora_files if f["stage"]],
            "sample_keys": sample_keys,
            "previews": progress.previews,
            "steps_completed": progress.total,
            "train_seconds": train_seconds,
            "total_seconds": int(time.time() - started),
            "warmup_seconds": int((progress.train_started or started) - started),
            "sec_per_step": progress.sec_per_step(),
            "final_loss": progress.loss,
            "gpu": gpu_name(),
            "dataset": ds_stats,
            "hf_home": env["HF_HOME"],
        }
        log(f"done: {json.dumps(result)}")
        if REFRESH_WORKER:
            result["refresh_worker"] = True
        return result
    except Exception as exc:
        traceback.print_exc()
        return {"error": f"Unexpected trainer error: {exc}", "code": "internal", "log_tail": list(trainer.tail) if trainer else []}
    finally:
        if trainer:
            trainer.terminate(grace_s=5)
        if workdir:
            shutil.rmtree(workdir, ignore_errors=True)  # datasets are personal — never keep them on the worker


if __name__ == "__main__":
    log(f"starting: ai_toolkit={AI_TOOLKIT_DIR} commit={os.environ.get('AI_TOOLKIT_COMMIT', '?')} "
        f"hf_home={hf_home()} gpu={gpu_name()}")
    runpod.serverless.start({"handler": handler})
