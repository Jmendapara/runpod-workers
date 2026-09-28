"""Verify the Qwen3-TTS stack at build time. Non-zero exit fails the build."""
import os
import sys


def main() -> int:
    import torch
    import torchaudio
    import transformers
    import qwen_tts  # noqa: F401
    from qwen_tts import Qwen3TTSModel  # noqa: F401

    root = "/comfyui/models/qwen3-tts"
    for name in ("Qwen3-TTS-12Hz-1.7B-Base", "Qwen3-TTS-12Hz-0.6B-Base", "Qwen3-TTS-12Hz-1.7B-VoiceDesign"):
        for rel in ("config.json", "model.safetensors", "speech_tokenizer/model.safetensors"):
            p = os.path.join(root, name, rel)
            if not os.path.isfile(p):
                print(f"FATAL: missing {p}", file=sys.stderr)
                return 1
    if not os.path.isfile("/comfyui/custom_nodes/pd_qwen3_tts/__init__.py"):
        print("FATAL: pd_qwen3_tts node not installed", file=sys.stderr)
        return 1
    print("Verification OK: qwen_tts importable, weights + node present")
    print(f"  torch={torch.__version__} CUDA={torch.version.cuda} torchaudio={torchaudio.__version__} transformers={transformers.__version__}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
