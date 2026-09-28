"""post_install: copy the in-repo PD node pack into ComfyUI's custom_nodes."""
import shutil
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent / "nodes" / "pd_qwen3_tts"
DST = Path("/comfyui/custom_nodes/pd_qwen3_tts")


def main() -> int:
    if not SRC.is_dir():
        print(f"FATAL: node source missing: {SRC}", file=sys.stderr)
        return 1
    if DST.exists():
        shutil.rmtree(DST)
    shutil.copytree(SRC, DST)
    print(f"Installed {SRC.name} -> {DST}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
