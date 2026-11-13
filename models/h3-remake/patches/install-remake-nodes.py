#!/usr/bin/env python3
"""post_install: copy the vendored Viggle Remake node pack into /comfyui/custom_nodes.

The pack (RemakeConditioning, RemakeKeyframeUntil, RemakeNoise, RemakePrefix) is published only
inside the Hugging Face model repo Viggle/Viggle-Remake (comfyui/ComfyUI-Viggle-Remake), not as a
git repo, so `custom_nodes:` cannot clone it. The two files under nodes/ are that folder at HF
revision 7e5be143702dd53b9154482de673b8519ed2d4d6 (2026-11-07), unmodified, Apache-2.0
(nodes/ComfyUI-Viggle-Remake/LICENSE). Their hashes are checked here so an edit cannot slip in.
"""
import hashlib
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "nodes", "ComfyUI-Viggle-Remake")
DST = "/comfyui/custom_nodes/ComfyUI-Viggle-Remake"
SHA256 = {
    "__init__.py": "4c0f96b0f0006df5a559d09d44b8d521a32d7992383223310b5ba2a3f88e3e42",
    "nodes.py": "13de28936065112503bc30abefefb356bfb3d51709a51ce28993e8b42001645c",
}


def main():
    for name, want in SHA256.items():
        with open(os.path.join(SRC, name), "rb") as f:
            got = hashlib.sha256(f.read()).hexdigest()
        if got != want:
            sys.exit(f"FATAL: {name} sha256 {got} != {want} (the vendored node pack was changed)")
    if os.path.exists(DST):
        shutil.rmtree(DST)
    shutil.copytree(SRC, DST)
    print(f"==> Installed ComfyUI-Viggle-Remake into {DST}: {sorted(os.listdir(DST))}", flush=True)


if __name__ == "__main__":
    main()
