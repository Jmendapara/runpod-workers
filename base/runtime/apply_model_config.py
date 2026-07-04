#!/usr/bin/env python3
"""Apply model.yaml at Docker build time.

Called as: apply_model_config.py <model.yaml> <model-ctx> [--phase P] [--shard K/N]

Phases (default --phase all does everything in one process, i.e. one layer):
  setup     — merge extra_model_paths, clone custom_nodes + pip, pip_extras.
              Cleans the pip/uv caches so they don't bloat the layer.
  download  — bake model weights into the image. With --shard K/N, only the
              downloads where (index %% N == K) are fetched, so a build can
              split weights across N separate RUN steps → N separate image
              layers. This matters because a single monolithic ~150 GB layer
              cannot be pushed to Docker Hub within its blob upload-session
              window; per-shard layers each fit and failed pushes resume
              blob-by-blob. Cleans the HF cache so it stays out of the layer.
  finalize  — run post_install scripts, write env, print manifest.

Run order for a sharded build: setup → download(0/N)…download(N-1/N) → finalize.

Fails the build on any error.
"""
from __future__ import annotations

import argparse
import os
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, "/opt/worker")

from lib.config import load_model_config  # noqa: E402

COMFY_CUSTOM_NODES = Path("/comfyui/custom_nodes")
COMFY_EXTRA_PATHS = Path("/comfyui/extra_model_paths.yaml")
WORKER_ENV_FILE = Path("/etc/worker/env")
PIP = "/opt/venv/bin/pip"


def run(cmd: list[str], cwd: str | None = None) -> None:
    """Run a command, stream output, fail loud on non-zero exit."""
    print(f"+ {' '.join(shlex.quote(c) for c in cmd)}", flush=True)
    result = subprocess.run(cmd, cwd=cwd, check=False)
    if result.returncode != 0:
        print(f"FATAL: command failed (exit {result.returncode}): {' '.join(cmd)}", file=sys.stderr)
        sys.exit(result.returncode)


def parse_shard(shard: str | None) -> tuple[int, int]:
    """Parse a "K/N" shard spec. None means a single shard covering everything."""
    if not shard:
        return 0, 1
    try:
        k_str, n_str = shard.split("/", 1)
        k, n = int(k_str), int(n_str)
    except ValueError:
        print(f"FATAL: --shard must be K/N (got {shard!r})", file=sys.stderr)
        sys.exit(64)
    if n <= 0 or not (0 <= k < n):
        print(f"FATAL: --shard K/N requires 0 <= K < N and N > 0 (got {shard!r})", file=sys.stderr)
        sys.exit(64)
    return k, n


def setup_phase(cfg) -> None:
    COMFY_CUSTOM_NODES.mkdir(parents=True, exist_ok=True)

    # 1. Merge extra_model_paths_additions
    if cfg.extra_model_paths_additions:
        print(f"==> Merging {len(cfg.extra_model_paths_additions)} extra_model_paths additions", flush=True)
        import yaml
        with open(COMFY_EXTRA_PATHS) as f:
            current = yaml.safe_load(f) or {}
        section = current.setdefault("runpod_worker_comfy", {})
        for k, v in cfg.extra_model_paths_additions.items():
            section[k] = v
            print(f"    {k}: {v}", flush=True)
        with open(COMFY_EXTRA_PATHS, "w") as f:
            yaml.safe_dump(current, f, default_flow_style=False, sort_keys=False)

    # 2. Custom nodes
    manifest_nodes: list[tuple[str, str]] = []
    for node in cfg.custom_nodes:
        repo = node.repo
        name = os.path.basename(repo).removesuffix(".git")
        target = COMFY_CUSTOM_NODES / name
        if target.exists():
            print(f"==> Skipping clone (already exists): {target}", flush=True)
        else:
            ref = node.ref
            # `git clone --branch` accepts branches/tags but NOT commit SHAs —
            # for a SHA pin we need a full clone, then detach at the commit.
            is_sha = bool(ref and re.fullmatch(r"[0-9a-fA-F]{7,40}", ref))
            if is_sha:
                print(f"==> Cloning {repo} (pinned commit {ref})", flush=True)
                run(["git", "clone", repo, str(target)])
                run(["git", "-C", str(target), "checkout", "--detach", ref])
            else:
                clone_cmd = ["git", "clone", "--depth", "1"]
                if ref:
                    clone_cmd += ["--branch", ref]
                clone_cmd += [repo, str(target)]
                print(f"==> Cloning {repo}" + (f" @ {ref}" if ref else ""), flush=True)
                run(clone_cmd)

        sha = subprocess.run(
            ["git", "-C", str(target), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True,
        ).stdout.strip() or "unknown"
        manifest_nodes.append((name, sha))

        req = target / "requirements.txt"
        if node.pip_install_requirements and req.exists():
            print(f"==> Installing {name}/requirements.txt", flush=True)
            run([PIP, "install", "-q", "--root-user-action=ignore", "-r", str(req)])

        for extra in node.extra_pip:
            print(f"==> Extra pip for {name}: {extra}", flush=True)
            run([PIP, "install", "-q", "--root-user-action=ignore", *shlex.split(extra)])

    # 3. Model-level pip extras
    for extra in cfg.pip_extras:
        print(f"==> Model pip_extras: {extra}", flush=True)
        run([PIP, "install", "-q", "--root-user-action=ignore", *shlex.split(extra)])

    # Keep pip/uv caches out of this layer.
    for cache in ("/root/.cache/pip", "/root/.cache/uv"):
        shutil.rmtree(cache, ignore_errors=True)
    subprocess.run(["uv", "cache", "clean"], check=False)

    print("==> setup manifest:", flush=True)
    print(f"    model: {cfg.name}", flush=True)
    if manifest_nodes:
        print(f"    custom_nodes ({len(manifest_nodes)}):", flush=True)
        for name, sha in manifest_nodes:
            print(f"      • {name} @ {sha}", flush=True)
    if cfg.pip_extras:
        print(f"    pip_extras: {list(cfg.pip_extras)}", flush=True)


def download_phase(cfg, shard_k: int, shard_n: int) -> None:
    if not cfg.model_downloads:
        print("==> No model_downloads to process.", flush=True)
        return

    hf_token = os.environ.get("HUGGINGFACE_ACCESS_TOKEN") or None
    manifest_downloads: list[str] = []

    for index, dl in enumerate(cfg.model_downloads):
        if index % shard_n != shard_k:
            continue
        dest_dir = Path(dl.dest)
        dest_dir.mkdir(parents=True, exist_ok=True)

        if dl.source == "hf":
            from huggingface_hub import hf_hub_download
            print(f"==> [{index}] HF download: {dl.repo_id}/{dl.filename} -> {dest_dir}", flush=True)
            tmp = Path("/tmp/hf-stage")
            tmp.mkdir(parents=True, exist_ok=True)
            got = hf_hub_download(
                repo_id=dl.repo_id,
                filename=dl.filename,
                local_dir=str(tmp),
                token=hf_token,
            )
            final_name = dl.rename or os.path.basename(dl.filename)
            final_path = dest_dir / final_name
            if str(Path(got).resolve()) != str(final_path.resolve()):
                Path(got).replace(final_path)
            shutil.rmtree(tmp, ignore_errors=True)
            manifest_downloads.append(f"hf {dl.repo_id}/{dl.filename} -> {final_path}")

        elif dl.source == "hf-snapshot":
            from huggingface_hub import snapshot_download
            print(f"==> [{index}] HF snapshot: {dl.repo_id} -> {dest_dir}", flush=True)
            snapshot_download(
                repo_id=dl.repo_id,
                local_dir=str(dest_dir),
                token=hf_token,
            )
            manifest_downloads.append(f"hf-snapshot {dl.repo_id} -> {dest_dir}")

        elif dl.source == "url":
            final_path = dest_dir / dl.filename
            print(f"==> [{index}] URL download: {dl.url} -> {final_path}", flush=True)
            wget_cmd = ["wget", "-q", "--show-progress", "-O", str(final_path)]
            if dl.auth_header_env:
                token = os.environ.get(dl.auth_header_env)
                if not token:
                    print(
                        f"FATAL: model_downloads requires env var {dl.auth_header_env}, which is unset",
                        file=sys.stderr,
                    )
                    sys.exit(2)
                wget_cmd += ["--header", f"Authorization: Bearer {token}"]
            wget_cmd.append(dl.url)
            run(wget_cmd)
            manifest_downloads.append(f"url {dl.url} -> {final_path}")

        else:
            print(f"FATAL: unknown model_downloads source: {dl.source!r}", file=sys.stderr)
            sys.exit(2)

    # Keep the HF cache out of this shard's layer.
    shutil.rmtree("/root/.cache/huggingface", ignore_errors=True)

    shard_desc = f"{shard_k}/{shard_n}" if shard_n > 1 else "all"
    print(f"==> download shard {shard_desc}: {len(manifest_downloads)} file(s)", flush=True)
    for line in manifest_downloads:
        print(f"      • {line}", flush=True)


def finalize_phase(cfg, ctx_dir: Path) -> None:
    # Post-install scripts (run after all weights are present).
    for script in cfg.post_install:
        script_abs = (ctx_dir / script).resolve()
        if not script_abs.exists():
            print(f"FATAL: post_install script not found: {script_abs}", file=sys.stderr)
            sys.exit(2)
        print(f"==> Running post_install: {script_abs}", flush=True)
        run([sys.executable, str(script_abs)])

    # env: write to /etc/worker/env, sourced by start.sh
    if cfg.env:
        WORKER_ENV_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(WORKER_ENV_FILE, "w") as f:
            for k, v in cfg.env.items():
                f.write(f"export {k}={shlex.quote(v)}\n")
        print(f"==> Wrote {len(cfg.env)} env var(s) to {WORKER_ENV_FILE}", flush=True)

    print("==> finalize manifest:", flush=True)
    print(f"    model: {cfg.name}", flush=True)
    print(f"    output.type: {cfg.output.type}", flush=True)
    if cfg.post_install:
        print(f"    post_install: {list(cfg.post_install)}", flush=True)
    print("==> apply_model_config.py complete.", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply model.yaml at Docker build time.")
    parser.add_argument("config", help="path to model.yaml")
    parser.add_argument("ctx", help="model build context dir (for post_install scripts)")
    parser.add_argument(
        "--phase",
        choices=["all", "setup", "download", "finalize"],
        default="all",
        help="which build phase to run (default: all)",
    )
    parser.add_argument(
        "--shard",
        default=None,
        metavar="K/N",
        help="with --phase download/all: only fetch downloads where index %% N == K",
    )
    args = parser.parse_args()

    ctx_dir = Path(args.ctx).resolve()

    print(f"==> Loading model config from {args.config}", flush=True)
    cfg = load_model_config(args.config)
    shard_k, shard_n = parse_shard(args.shard)
    print(
        f"==> Model: {cfg.name} (output={cfg.output.type}) "
        f"phase={args.phase} shard={shard_k}/{shard_n}",
        flush=True,
    )

    if args.phase in ("all", "setup"):
        setup_phase(cfg)
    if args.phase in ("all", "download"):
        download_phase(cfg, shard_k, shard_n)
    if args.phase in ("all", "finalize"):
        finalize_phase(cfg, ctx_dir)


if __name__ == "__main__":
    main()
