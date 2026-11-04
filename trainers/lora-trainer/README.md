# lora-trainer

RunPod serverless worker that trains a LoRA with [ostris/ai-toolkit](https://github.com/ostris/ai-toolkit)
for OpenSourceGen's **Train** tab (`opensourcegen/server/handlers/loraTraining.js`). One image for every
supported base model (FLUX.1, FLUX.2 klein, Chroma, Qwen Image 2.1, Z-Image Turbo, Krea 2, SDXL, SD 1.5,
Wan 2.2 14B/5B, LTX-2.3, MiniMax H3, …): **no weights are baked**. Base models are downloaded from Hugging
Face on first use into `HF_HOME`, which is the endpoint's network volume (`/runpod-volume/hf`) when one is
attached, else the container disk (`/workspace/hf`, lost when the worker is recycled).

Not a `model.yaml` worker — it has its own `Dockerfile` (CUDA 13.0.3 runtime + torch 2.13 cu130 + ai-toolkit
pinned by `AI_TOOLKIT_COMMIT`). Build from the repo root:

```bash
./tools/remote-build.sh lora-trainer      # Hetzner box (recommended)
MODEL=lora-trainer bash build.sh          # any docker host with buildx
```

Tag pushed: `jmendapara/lora-trainer-runpod-worker:YYYY-MM-DD-HHMM-<sha>`.

## Endpoint settings

- GPU pools: 80 GB class first (H100 80GB, RTX PRO 6000 96GB, A100 80GB). Every catalog model is sized
  for 80 GB with fp8 / int8 weights; the 14B video models need it.
- Container disk: 200 GB when NO network volume is attached (the HF cache lives there); 60 GB with one.
- Network volume: 300–500 GB STANDARD, `/runpod-volume`. Pins the endpoint to the volume's data center.
- Workers min 0 / max 2–3. FlashBoot off (the image is ~25 GB and there is nothing to keep warm).
- Execution timeout: the server sends `policy.executionTimeout` per job (3× the estimate + 30 min, capped at
  6 h); set the endpoint default to ≥ 3 h as a backstop.
- `minCudaVersion` 13.0 (torch cu130).
- Env: only the four R2 vars (`BUCKET_ENDPOINT_URL`, `BUCKET_ACCESS_KEY_ID`, `BUCKET_SECRET_ACCESS_KEY`,
  `R2_BUCKET_NAME`), optionally `R2_INPUT_BUCKET_NAME` / `BUCKET_PROFILES` (same contract as the ComfyUI
  workers). Hugging Face tokens for gated models (FLUX.1-dev, Krea 2) arrive **per job** in `input.hf_token`.
- `opensourcegen/server/scripts/setup-lora-trainer-endpoint.js` creates all of this through RunPod's v2 API.

## Job contract

Input (`input`):

```jsonc
{
  "op": "train",
  "training_id": "tr_…", "uid": "firebase-uid", "name": "My Dog Rex", "trigger_word": "rexdog",
  "control_url": "https://…/api/lora-training-control/tr_…?token=…",   // polled every 20 s; {cancel:true} stops the run
  "model": { "arch": "zimage", "name_or_path": "Tongyi-MAI/Z-Image-Turbo", "quantize": true, "qtype": "qfloat8",
             "quantize_te": true, "qtype_te": "qfloat8", "low_vram": false,
             "assistant_lora_path": "ostris/zimage_turbo_training_adapter/…", "model_kwargs": {} },
  "dataset": { "items": [{ "r2_key": "users/u/lora-datasets/x.jpg", "kind": "image", "caption": "…" }],
               "default_caption": "rexdog", "resolution": 1024, "caption_dropout_rate": 0.05,
               "num_frames": 33, "fps": 16, "do_i2v": false, "do_audio": false },          // video models only
  "train": { "steps": 1000, "lr": 0.0001, "rank": 16, "alpha": 16, "timestep_type": "weighted",
             "noise_scheduler": "flowmatch", "cache_text_embeddings": false, "optimizer": "adamw8bit" },
  "network": { "ignore_if_contains": ["adaln_proj"] },
  "sample": { "enabled": true, "prompts": ["rexdog, portrait photo"], "width": 1024, "height": 1024,
              "sample_steps": 9, "guidance_scale": 1, "sampler": "flowmatch" },
  "output": { "prefix": "users/u/loras/tr_…/", "filename": "my-dog-rex.safetensors" },
  "hf_token": "hf_…",            // optional, gated models
  "bucket_profile": "osg-qa"     // optional
}
```

Progress (`runpod.serverless.progress_update`, visible as `output` on `/status` while IN_PROGRESS):
`{ phase: preparing|loading|caching|training|sampling|uploading, step, total_steps, pct, elapsed_s, eta_s,
sec_per_step, loss, message, worker_id, gpu, previews[] }`.

Result: `{ lora_key, lora_bytes, sample_keys[], steps_completed, train_seconds, total_seconds, warmup_seconds,
sec_per_step, final_loss, gpu, dataset }`, or `{ cancelled: true, steps_completed, train_seconds }`, or
`{ error, code: invalid_input|storage_not_configured|dataset_download_failed|training_failed|oom|no_output|bad_lora|internal,
log_tail[] }`. Every result also carries `worker_id`, `gpu`, `phase`, `total_seconds` and, once storage is reachable, `log_key`.

Output layout in R2: `<prefix><filename>` (the LoRA, fp16 safetensors), `<prefix>samples/<n>.jpg` (final-step
previews, image models only), `<prefix>config.yaml` (the exact ai-toolkit config, for support), `<prefix>logs/run.log` (the run's whole log, uploaded
every 2 minutes and at the end, with the HF token and control URL redacted, so it survives a crash and the worker).

## Tests

```bash
python3 tests/test_lora_trainer.py
```
