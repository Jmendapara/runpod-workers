"""Recreate the NF4-v2 -> NF4 symlink the standalone worker made at build time.

The model is downloaded into /comfyui/models/HunyuanImage-3.0-Instruct-Distil-NF4
(bare name), but the custom node / workflow may reference the repo's -v2 name.
A relative symlink keeps both paths resolvable, matching the original Dockerfile.
Runs at build time; non-zero exit fails the build.
"""

import os
import sys

MODELS = "/comfyui/models"
REAL = "HunyuanImage-3.0-Instruct-Distil-NF4"
LINK = os.path.join(MODELS, "HunyuanImage-3.0-Instruct-Distil-NF4-v2")

real_path = os.path.join(MODELS, REAL)
if not os.path.isdir(real_path):
    print(f"WARNING: {real_path} not found — skipping symlink", file=sys.stderr)
    sys.exit(0)

if os.path.islink(LINK) or os.path.exists(LINK):
    print(f"Symlink target already exists: {LINK}")
else:
    os.symlink(REAL, LINK)  # relative target, mirrors `ln -s REAL LINK`
    print(f"Created symlink {LINK} -> {REAL}")
