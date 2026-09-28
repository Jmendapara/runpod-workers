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
import hashlib
import os
import random
import re
import threading
import time
import wave
from collections import OrderedDict

import numpy as np
import torch

import folder_paths

MODEL_ROOT = os.path.join(folder_paths.models_dir, "qwen3-tts")
MODEL_DIRS = {
    "base": "Qwen3-TTS-12Hz-1.7B-Base",
    "base_small": "Qwen3-TTS-12Hz-0.6B-Base",
    "voice_design": "Qwen3-TTS-12Hz-1.7B-VoiceDesign",
}
CLONE_SIZES = {"1.7B": "base", "0.6B": "base_small"}

if torch.cuda.is_available():
    # Faster matmuls on Ampere+ (TF32) — speech quality is unaffected.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# Voice clone prompts (the encoded reference clip + transcript) per model: every voice note of a
# companion clones the SAME clip, so it is encoded once per worker, not once per note.
_PROMPT_CACHE_MAX = 64
_prompt_cache = OrderedDict()
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


_SENTENCE_END = re.compile(r"(?<=[.!?…])\s+")


def _chunks(text, target_chars):
    """Split text at sentence ends into groups of about target_chars (one group when short)."""
    text = " ".join(str(text).split())
    if target_chars <= 0 or len(text) <= target_chars * 1.3:
        return [text]
    out, cur = [], ""
    for sentence in _SENTENCE_END.split(text):
        if cur and len(cur) + 1 + len(sentence) > target_chars:
            out.append(cur)
            cur = sentence
        else:
            cur = f"{cur} {sentence}".strip()
    if cur:
        out.append(cur)
    return out


def _join(wavs, sr, pause_s=0.22):
    """Concatenate chunks with a short breath of silence (trailing silence of each chunk trimmed)."""
    gap = np.zeros(int(sr * pause_s), dtype=np.float32)
    parts = []
    for i, w in enumerate(wavs):
        w = np.asarray(w, dtype=np.float32).reshape(-1)
        nz = np.flatnonzero(np.abs(w) > 1e-3)
        if nz.size:
            w = w[: nz[-1] + int(sr * 0.05)]
        if i:
            parts.append(gap)
        parts.append(w)
    return np.concatenate(parts) if parts else np.zeros(1, dtype=np.float32)


def _clone_prompt(model, kind, wav, sr, ref_text, x_vector_only):
    h = hashlib.sha1()
    h.update(kind.encode())
    h.update(np.ascontiguousarray(wav, dtype=np.float32).tobytes())
    h.update(str(sr).encode())
    h.update((ref_text or "").encode("utf-8"))
    h.update(b"x" if x_vector_only else b"i")
    key = h.hexdigest()
    with _lock:
        hit = _prompt_cache.get(key)
        if hit is not None:
            _prompt_cache.move_to_end(key)
            return hit, True
    items = model.create_voice_clone_prompt(
        ref_audio=(wav, sr), ref_text=None if x_vector_only else ref_text, x_vector_only_mode=x_vector_only)
    with _lock:
        _prompt_cache[key] = items
        while len(_prompt_cache) > _PROMPT_CACHE_MAX:
            _prompt_cache.popitem(last=False)
    return items, False


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
        }, "optional": {
            "model_size": (list(CLONE_SIZES.keys()), {"default": "1.7B"}),
            "non_streaming": ("BOOLEAN", {"default": True}),
            # > 0: long text is split at sentence ends into groups of about this many characters,
            # generated as ONE batch (same reference, same seed) and joined — a long voice note
            # takes about as long as its longest group.
            "chunk_chars": ("INT", {"default": 0, "min": 0, "max": 2000}),
        }}

    def run(self, ref_audio, ref_text, text, language, x_vector_only, seed, temperature, top_p, top_k,
            repetition_penalty, max_new_tokens, model_size="1.7B", non_streaming=True, chunk_chars=0):
        if not str(text).strip():
            raise ValueError("text is empty")
        ref_text = str(ref_text or "").strip()
        # ICL cloning needs the reference transcript; without one fall back to the speaker
        # embedding alone (same timbre, slightly less faithful prosody).
        x_vector_only = bool(x_vector_only) or not ref_text
        kind = CLONE_SIZES.get(model_size, "base")
        model = _load(kind)
        wav, sr = _from_audio(ref_audio)
        t0 = time.time()
        with torch.inference_mode():
            prompt, cached = _clone_prompt(model, kind, wav, sr, ref_text, x_vector_only)
            t1 = time.time()
            parts = _chunks(text, int(chunk_chars or 0))
            _seed_everything(seed)
            wavs, out_sr = model.generate_voice_clone(
                text=parts if len(parts) > 1 else parts[0],
                language=[language] * len(parts) if len(parts) > 1 else language,
                voice_clone_prompt=list(prompt) * len(parts) if len(parts) > 1 else prompt,
                non_streaming_mode=bool(non_streaming),
                **_gen_kwargs(temperature, top_p, top_k, repetition_penalty, max_new_tokens))
            if len(parts) > 1:
                wavs = [_join(wavs, out_sr)]
        t2 = time.time()
        secs = len(np.asarray(wavs[0]).reshape(-1)) / float(out_sr or 1)
        print(f"[PDQwen3TTS] clone {model_size} x{len(parts)}: prompt {'cached' if cached else f'{t1 - t0:.1f}s'}, "
              f"speech {secs:.1f}s in {t2 - t1:.1f}s (RTF {(t2 - t1) / max(secs, 0.1):.2f})", flush=True)
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
