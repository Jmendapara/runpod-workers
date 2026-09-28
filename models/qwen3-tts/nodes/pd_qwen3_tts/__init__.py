"""Pixel Dream Qwen3-TTS nodes (Apache-2.0 weights: Qwen/Qwen3-TTS-12Hz-1.7B-{Base,VoiceDesign}).

Three nodes, kept deliberately small so every knob the app relies on is explicit:

  PDQwen3TTSVoiceDesign  text + natural-language voice description -> AUDIO
                         (used once per companion to create her reference clip)
  PDQwen3TTSVoiceClone   reference AUDIO + its exact transcript + text -> AUDIO
                         (every voice note: same clip + transcript = same voice)
  PDSaveWav              AUDIO -> 16-bit PCM mono WAV at the model's native rate (24 kHz)
                         under ComfyUI's output dir, reported as an "audio" output

Weights are baked into the image under models/qwen3-tts/<repo name>/ (no runtime download:
HF_HUB_OFFLINE=1). Models load lazily on first use and stay resident.
"""
import os
import random
import threading
import wave

import numpy as np
import torch

import folder_paths

MODEL_ROOT = os.path.join(folder_paths.models_dir, "qwen3-tts")
MODEL_DIRS = {
    "base": "Qwen3-TTS-12Hz-1.7B-Base",
    "voice_design": "Qwen3-TTS-12Hz-1.7B-VoiceDesign",
}
LANGUAGES = ["Auto", "English", "Chinese", "Japanese", "Korean", "German", "French",
             "Russian", "Portuguese", "Spanish", "Italian"]
# 12 Hz codec: 12 tokens per second of speech. 1500 tokens = 125 s, far above any voice note;
# it only bounds a runaway generation.
MAX_NEW_TOKENS_CAP = 1500

_models = {}
_lock = threading.Lock()


def _attn_impl():
    try:
        import flash_attn  # noqa: F401
        return "flash_attention_2"
    except Exception:
        return "sdpa"


def _load(kind):
    with _lock:
        model = _models.get(kind)
        if model is None:
            from qwen_tts import Qwen3TTSModel
            path = os.path.join(MODEL_ROOT, MODEL_DIRS[kind])
            if not os.path.isdir(path):
                raise RuntimeError(f"Qwen3-TTS weights missing: {path}")
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
            print(f"[PDQwen3TTS] loading {kind} from {path} on {device} ({_attn_impl()})", flush=True)
            model = Qwen3TTSModel.from_pretrained(path, device_map=device, dtype=dtype,
                                                  attn_implementation=_attn_impl())
            _models[kind] = model
        return model


def _seed_everything(seed):
    seed = int(seed) & 0xFFFFFFFF
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _gen_kwargs(temperature, top_p, top_k, repetition_penalty, max_new_tokens):
    return {
        "do_sample": True,
        "temperature": float(temperature),
        "top_p": float(top_p),
        "top_k": int(top_k),
        "repetition_penalty": float(repetition_penalty),
        "subtalker_dosample": True,
        "subtalker_temperature": float(temperature),
        "subtalker_top_p": float(top_p),
        "subtalker_top_k": int(top_k),
        "max_new_tokens": min(int(max_new_tokens), MAX_NEW_TOKENS_CAP),
    }


def _to_audio(wavs, sr):
    wav = np.asarray(wavs[0], dtype=np.float32).reshape(-1)
    wav = np.clip(wav, -1.0, 1.0)
    return {"waveform": torch.from_numpy(wav).unsqueeze(0).unsqueeze(0), "sample_rate": int(sr)}


def _from_audio(audio):
    waveform = audio["waveform"]
    if isinstance(waveform, torch.Tensor):
        waveform = waveform.detach().float().cpu().numpy()
    waveform = np.asarray(waveform, dtype=np.float32)
    if waveform.ndim == 3:
        waveform = waveform[0]
    if waveform.ndim == 2:
        waveform = waveform.mean(axis=0)
    return waveform.reshape(-1), int(audio["sample_rate"])


def _sampling_inputs():
    return {
        "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFF}),
        "temperature": ("FLOAT", {"default": 0.9, "min": 0.1, "max": 1.5, "step": 0.05}),
        "top_p": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 1.0, "step": 0.05}),
        "top_k": ("INT", {"default": 50, "min": 1, "max": 200}),
        "repetition_penalty": ("FLOAT", {"default": 1.05, "min": 1.0, "max": 2.0, "step": 0.01}),
        "max_new_tokens": ("INT", {"default": 1200, "min": 16, "max": MAX_NEW_TOKENS_CAP}),
    }


class PDQwen3TTSVoiceDesign:
    CATEGORY = "audio/qwen3-tts"
    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "text": ("STRING", {"multiline": True, "default": ""}),
            "instruct": ("STRING", {"multiline": True, "default": ""}),
            "language": (LANGUAGES, {"default": "English"}),
            **_sampling_inputs(),
        }}

    def run(self, text, instruct, language, seed, temperature, top_p, top_k, repetition_penalty, max_new_tokens):
        if not str(text).strip():
            raise ValueError("text is empty")
        model = _load("voice_design")
        _seed_everything(seed)
        with torch.inference_mode():
            wavs, sr = model.generate_voice_design(
                text=str(text), instruct=str(instruct or ""), language=language,
                **_gen_kwargs(temperature, top_p, top_k, repetition_penalty, max_new_tokens))
        return (_to_audio(wavs, sr),)


class PDQwen3TTSVoiceClone:
    CATEGORY = "audio/qwen3-tts"
    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "ref_audio": ("AUDIO",),
            "ref_text": ("STRING", {"multiline": True, "default": ""}),
            "text": ("STRING", {"multiline": True, "default": ""}),
            "language": (LANGUAGES, {"default": "English"}),
            "x_vector_only": ("BOOLEAN", {"default": False}),
            **_sampling_inputs(),
        }}

    def run(self, ref_audio, ref_text, text, language, x_vector_only, seed, temperature, top_p, top_k,
            repetition_penalty, max_new_tokens):
        if not str(text).strip():
            raise ValueError("text is empty")
        ref_text = str(ref_text or "").strip()
        # ICL cloning needs the reference transcript; without one fall back to the speaker
        # embedding alone (same timbre, slightly less faithful prosody).
        x_vector_only = bool(x_vector_only) or not ref_text
        model = _load("base")
        wav, sr = _from_audio(ref_audio)
        _seed_everything(seed)
        with torch.inference_mode():
            wavs, out_sr = model.generate_voice_clone(
                text=str(text), language=language, ref_audio=(wav, sr),
                ref_text=None if x_vector_only else ref_text, x_vector_only_mode=x_vector_only,
                **_gen_kwargs(temperature, top_p, top_k, repetition_penalty, max_new_tokens))
        return (_to_audio(wavs, out_sr),)


class PDSaveWav:
    CATEGORY = "audio/qwen3-tts"
    RETURN_TYPES = ()
    FUNCTION = "save"
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "audio": ("AUDIO",),
            "filename_prefix": ("STRING", {"default": "PD_voice"}),
        }}

    def save(self, audio, filename_prefix):
        wav, sr = _from_audio(audio)
        pcm = (np.clip(wav, -1.0, 1.0) * 32767.0).astype("<i2")
        out_dir = folder_paths.get_output_directory()
        full_dir, filename, counter, subfolder, _ = folder_paths.get_save_image_path(filename_prefix, out_dir)
        name = f"{filename}_{counter:05}_.wav"
        with wave.open(os.path.join(full_dir, name), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sr)
            w.writeframes(pcm.tobytes())
        return {"ui": {"audio": [{"filename": name, "subfolder": subfolder, "type": "output"}]}}


NODE_CLASS_MAPPINGS = {
    "PDQwen3TTSVoiceDesign": PDQwen3TTSVoiceDesign,
    "PDQwen3TTSVoiceClone": PDQwen3TTSVoiceClone,
    "PDSaveWav": PDSaveWav,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "PDQwen3TTSVoiceDesign": "Qwen3-TTS Voice Design (PD)",
    "PDQwen3TTSVoiceClone": "Qwen3-TTS Voice Clone (PD)",
    "PDSaveWav": "Save WAV (PD)",
}
