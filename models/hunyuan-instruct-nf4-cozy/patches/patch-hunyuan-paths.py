"""Append RunPod network-volume search paths to hunyuan_shared.py.

Runs once at Docker build time so the custom node always includes
/runpod-volume/models in its folder_paths search list.
"""

import glob
import os
import sys

SEARCH_PATTERNS = [
    "/comfyui/custom_nodes/*/hunyuan_shared.py",
    "/comfyui/custom_nodes/hunyuan_shared.py",
]

PATCH = """

# --- RunPod network-volume search paths (injected by worker build) ---
try:
    _folder_paths.add_model_folder_path(HUNYUAN_FOLDER_NAME, "/runpod-volume/models")
    _folder_paths.add_model_folder_path(HUNYUAN_INSTRUCT_FOLDER_NAME, "/runpod-volume/models")
except Exception:
    pass
"""

# Find the file
targets = []
for pattern in SEARCH_PATTERNS:
    targets.extend(glob.glob(pattern))

if not targets:
    # List what's actually in custom_nodes so the build log is useful
    cn_dir = "/comfyui/custom_nodes"
    if os.path.isdir(cn_dir):
        print(f"Contents of {cn_dir}:")
        for item in sorted(os.listdir(cn_dir)):
            item_path = os.path.join(cn_dir, item)
            print(f"  {item}/ " if os.path.isdir(item_path) else f"  {item}")
            if os.path.isdir(item_path):
                for sub in sorted(os.listdir(item_path))[:10]:
                    print(f"    {sub}")
    else:
        print(f"{cn_dir} does not exist!")
    print("WARNING: hunyuan_shared.py not found — skipping patch", file=sys.stderr)
    sys.exit(0)

for target in targets:
    with open(target, "a") as f:
        f.write(PATCH)
    print(f"Patched {target}")
