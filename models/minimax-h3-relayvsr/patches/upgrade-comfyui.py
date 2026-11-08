#!/usr/bin/env python3
"""post_install: overlay a pinned ComfyUI master commit onto the base image's /comfyui.

RelayVSR's core nodes (RelayVSRConditioning / VSRDecode, Comfy-Org/ComfyUI#17511, merged
2026-11-05) are not in any tagged release yet (v0.44.0 is from 2026-11-04), and comfy-cli
only installs `latest` or `nightly`. Until a tagged release includes them, this swaps the
ComfyUI tree for the archive of one pinned commit and installs its requirements.txt
(comfy-kitchen 0.2.41 provides the window_attention kernel RelayVSR needs). Weights,
custom nodes, extra_model_paths.yaml and the input/output/user dirs are kept.

Remove this script (and build on a base whose ComfyUI >= the release that ships RelayVSR)
once such a release exists. Override the commit with COMFYUI_OVERLAY_SHA at build time.
"""
import os, shutil, subprocess, sys, tarfile, urllib.request

SHA = os.environ.get("COMFYUI_OVERLAY_SHA", "bdb0816a292d221efeb6ec4ae41bcdff3f91fc86")  # master 2026-11-08
COMFY = "/comfyui"
KEEP = ["custom_nodes", "models", "input", "output", "user", "temp", "extra_model_paths.yaml"]
PIP = "/opt/venv/bin/pip"
PY = "/opt/venv/bin/python"


def main():
    tgz = "/tmp/comfyui-overlay.tgz"
    url = f"https://github.com/comfyanonymous/ComfyUI/archive/{SHA}.tar.gz"
    print(f"==> Downloading ComfyUI {SHA[:8]} from {url}", flush=True)
    urllib.request.urlretrieve(url, tgz)
    with tarfile.open(tgz) as t:
        t.extractall("/tmp/comfyui-overlay", filter="data")
    src = f"/tmp/comfyui-overlay/ComfyUI-{SHA}"
    if not os.path.isdir(src):
        sys.exit(f"FATAL: extracted tree not found at {src}")

    old = COMFY + ".old"
    os.rename(COMFY, old)
    shutil.move(src, COMFY)
    for name in KEEP:
        s, d = os.path.join(old, name), os.path.join(COMFY, name)
        if not os.path.exists(s):
            continue
        if os.path.isdir(d) and not os.path.islink(d):
            shutil.rmtree(d)
        elif os.path.lexists(d):
            os.remove(d)
        shutil.move(s, d)
    shutil.rmtree(old)
    shutil.rmtree("/tmp/comfyui-overlay", ignore_errors=True)
    os.remove(tgz)

    print("==> Installing the overlay's requirements.txt", flush=True)
    subprocess.check_call([PIP, "install", "-q", "--root-user-action=ignore", "-r", f"{COMFY}/requirements.txt"])
    subprocess.call(["rm", "-rf", "/root/.cache/pip", "/root/.cache/uv"])

    # Verify: version module importable, the RelayVSR nodes present, comfy-kitchen has window_attention.
    code = (
        "import sys; sys.path.insert(0, '/comfyui'); import comfyui_version; print('ComfyUI', comfyui_version.__version__);"
        "import comfy_kitchen; print('comfy-kitchen', getattr(comfy_kitchen, '__version__', '?'));"
        "assert hasattr(comfy_kitchen, 'window_attention'), 'comfy_kitchen.window_attention missing';"
        "src = open('/comfyui/comfy_extras/nodes_wan.py').read();"
        "assert 'class RelayVSRConditioning' in src and 'class VSRDecode' in src, 'RelayVSR nodes missing';"
        "import os; assert os.path.exists('/comfyui/comfy/ldm/wan/model_vsr.py'); print('RelayVSR nodes present')"
    )
    subprocess.check_call([PY, "-c", code])
    print(f"==> ComfyUI overlay {SHA[:8]} applied.", flush=True)


if __name__ == "__main__":
    main()
