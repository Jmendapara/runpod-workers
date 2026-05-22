"""Verify omnivoice + critical imports at build time. Non-zero exit fails the build."""
import sys


def main() -> int:
    from PIL import Image  # noqa: F401
    import torch
    import omnivoice

    sys.path.insert(0, "/comfyui/custom_nodes/ComfyUI-OmniVoice-TTS")
    print("Verification OK: PIL, torch, omnivoice all importable")
    print(f"  torch={torch.__version__}, CUDA={torch.version.cuda}")
    print(f"  omnivoice={omnivoice.__version__}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
