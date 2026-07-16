"""Symlink /models -> /comfyui/models for CWD-relative model lookups.

ComfyUI-TranslationNode opens its M2M-100 weights via the relative path
"models/translation/facebook/m2m100_418M" (translation_node.py), which only
resolves if the process CWD is /comfyui. The container entrypoint runs with
CWD=/ (base/Dockerfile ends with WORKDIR /), so without this link the node
misses the baked weights and re-downloads ~2 GB from HuggingFace on the first
request of every cold worker.
"""

import os
import sys

LINK = "/models"
TARGET = "/comfyui/models"


def main() -> int:
    if os.path.islink(LINK):
        if os.readlink(LINK) == TARGET:
            print(f"symlink-models: {LINK} -> {TARGET} already present")
            return 0
        os.remove(LINK)
    elif os.path.exists(LINK):
        print(f"symlink-models: {LINK} exists and is not a symlink; refusing to replace", file=sys.stderr)
        return 1

    os.symlink(TARGET, LINK)
    print(f"symlink-models: created {LINK} -> {TARGET}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
