#!/usr/bin/env python3
"""Apply model.yaml at Docker build time.

Called as: apply_model_config.py /etc/worker/model.yaml /tmp/model-ctx

Steps:
1. Validate model.yaml against the schema.
2. Merge extra_model_paths_additions into /comfyui/extra_model_paths.yaml.
3. Clone each custom_nodes[] repo, pip-install requirements, then extra_pip.
4. Run pip_extras (model-level pip lines).
5. Run each post_install script as `python <abs path>` (non-zero fails the build).
6. Write env: into /etc/worker/env so start.sh can source it.
7. Print a manifest of everything installed/applied.

Fails the build on any error.
"""
from __future__ import annotations

import os
import shlex
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


def main() -> None:
    if len(sys.argv) != 3:
        print(f"usage: {sys.argv[0]} <model.yaml> <model-ctx>", file=sys.stderr)
        sys.exit(64)

    cfg_path = sys.argv[1]
    ctx_dir = Path(sys.argv[2]).resolve()

    print(f"==> Loading model config from {cfg_path}", flush=True)
    cfg = load_model_config(cfg_path)
    print(f"==> Model: {cfg.name} (output={cfg.output.type})", flush=True)

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
            clone_cmd = ["git", "clone", "--depth", "1"]
            if node.ref:
                clone_cmd += ["--branch", node.ref]
            clone_cmd += [repo, str(target)]
            print(f"==> Cloning {repo}", flush=True)
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

    # 3b. Model downloads (bake weights into the image)
    manifest_downloads: list[str] = []
    if cfg.model_downloads:
        hf_token = os.environ.get("HUGGINGFACE_ACCESS_TOKEN") or None
        for dl in cfg.model_downloads:
            dest_dir = Path(dl.dest)
            dest_dir.mkdir(parents=True, exist_ok=True)

            if dl.source == "hf":
                from huggingface_hub import hf_hub_download
                print(f"==> HF download: {dl.repo_id}/{dl.filename} -> {dest_dir}", flush=True)
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
                # Clean any subdirs hf_hub_download created under /tmp
                import shutil as _sh
                _sh.rmtree(tmp, ignore_errors=True)
                manifest_downloads.append(f"hf {dl.repo_id}/{dl.filename} -> {final_path}")

            elif dl.source == "hf-snapshot":
                from huggingface_hub import snapshot_download
                print(f"==> HF snapshot: {dl.repo_id} -> {dest_dir}", flush=True)
                snapshot_download(
                    repo_id=dl.repo_id,
                    local_dir=str(dest_dir),
                    token=hf_token,
                )
                manifest_downloads.append(f"hf-snapshot {dl.repo_id} -> {dest_dir}")

            elif dl.source == "url":
                final_path = dest_dir / dl.filename
                print(f"==> URL download: {dl.url} -> {final_path}", flush=True)
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

        # Reclaim cache space
        import shutil as _sh
        _sh.rmtree("/root/.cache/huggingface", ignore_errors=True)

    # 4. Post-install scripts
    for script in cfg.post_install:
        script_abs = (ctx_dir / script).resolve()
        if not script_abs.exists():
            print(f"FATAL: post_install script not found: {script_abs}", file=sys.stderr)
            sys.exit(2)
        print(f"==> Running post_install: {script_abs}", flush=True)
        run([sys.executable, str(script_abs)])

    # 5. env: write to /etc/worker/env, sourced by start.sh
    if cfg.env:
        WORKER_ENV_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(WORKER_ENV_FILE, "w") as f:
            for k, v in cfg.env.items():
                f.write(f"export {k}={shlex.quote(v)}\n")
        print(f"==> Wrote {len(cfg.env)} env var(s) to {WORKER_ENV_FILE}", flush=True)

    # 6. Manifest
    print("==> apply_model_config.py manifest:", flush=True)
    print(f"    model: {cfg.name}", flush=True)
    print(f"    output.type: {cfg.output.type}", flush=True)
    if manifest_nodes:
        print(f"    custom_nodes ({len(manifest_nodes)}):", flush=True)
        for name, sha in manifest_nodes:
            print(f"      • {name} @ {sha}", flush=True)
    if cfg.pip_extras:
        print(f"    pip_extras: {list(cfg.pip_extras)}", flush=True)
    if manifest_downloads:
        print(f"    model_downloads ({len(manifest_downloads)}):", flush=True)
        for line in manifest_downloads:
            print(f"      • {line}", flush=True)
    if cfg.post_install:
        print(f"    post_install: {list(cfg.post_install)}", flush=True)
    print("==> apply_model_config.py complete.", flush=True)


if __name__ == "__main__":
    main()
