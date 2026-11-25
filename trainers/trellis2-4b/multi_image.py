"""Several pictures of one object (front, back, a side…) → one 3D model.

TRELLIS.2 is trained on one picture and its pipeline has no multi-picture entry point. TRELLIS (v1)
shipped a tuning-free one, `run_multi_image`, and it carries over unchanged because the samplers are the
same flow-matching Euler samplers: every picture is encoded on its own, and at each sampler step the
model's prediction is steered by all of them.

  multidiffusion  the prediction is the mean of one prediction per picture, at every step. Costs one
                  extra model call per extra picture per step (the negative call is not repeated).
  stochastic      one picture per step, in turn, starting with the main picture. No extra cost, but with
                  12 steps and 4 pictures each one is seen three times.
  concat          the pictures' tokens joined into one conditioning sequence (not from TRELLIS; one model
                  call per step). Kept for comparison.

The mixing sits under classifier-free guidance, so the guidance interval and the CFG rescale of
pipeline.json apply to the mixed prediction exactly as they apply to a single picture's.

`make_inference` is pure Python (no torch), so the step logic is unit-tested on a laptop.
"""
from __future__ import annotations

from contextlib import contextmanager

CASCADE_RESOLUTION = {"1024_cascade": 1024, "1536_cascade": 1536}


def make_inference(original, count: int, mode: str):
    """A replacement for FlowEulerSampler._inference_model. `original` is that method; a conditioning
    whose batch is `count` is the pictures' (the negative conditioning has a batch of one and passes
    through untouched)."""
    state = {"step": 0}

    def inference(self, model, x_t, t, cond=None, **kwargs):
        if cond is None or cond.shape[0] != count:
            return original(self, model, x_t, t, cond, **kwargs)
        if mode == "stochastic":
            if t >= 1.0 - 1e-6:  # every sampler starts at t = 1: each stage begins with the main picture
                state["step"] = 0
            i = state["step"] % count
            state["step"] += 1
            return original(self, model, x_t, t, cond[i:i + 1], **kwargs)
        pred = original(self, model, x_t, t, cond[0:1], **kwargs)
        for i in range(1, count):
            pred = pred + original(self, model, x_t, t, cond[i:i + 1], **kwargs)
        return (1.0 / count) * pred

    return inference


@contextmanager
def several_pictures(sampler_cls, count: int, mode: str):
    """While open, every sampler built on `sampler_cls` mixes `count` pictures. One job runs at a time."""
    original = sampler_cls._inference_model
    sampler_cls._inference_model = make_inference(original, count, mode)
    try:
        yield
    finally:
        sampler_cls._inference_model = original


def conditioning(pipeline, pictures: list, resolution: int, mode: str) -> dict:
    import torch

    cond = pipeline.get_cond(pictures, resolution, include_neg_cond=False)["cond"]  # (pictures, tokens, dim)
    neg = torch.zeros_like(cond[:1])
    if mode == "concat":
        cond = cond.reshape(1, -1, cond.shape[-1])
    return {"cond": cond, "neg_cond": neg}


def run(pipeline, pictures: list, seed: int, pipeline_type: str, mode: str, max_num_tokens: int = 49152) -> list:
    """Trellis2ImageTo3DPipeline.run for the two cascade types, conditioned on several preprocessed
    pictures. Same stages, sampler settings and seeding as upstream."""
    import torch
    from trellis2.pipelines.samplers.flow_euler import FlowEulerSampler

    resolution = CASCADE_RESOLUTION[pipeline_type]
    models = pipeline.models
    with torch.no_grad():
        torch.manual_seed(seed)
        cond_512 = conditioning(pipeline, pictures, 512, mode)
        cond_1024 = conditioning(pipeline, pictures, 1024, mode)
        with several_pictures(FlowEulerSampler, len(pictures), mode):
            coords = pipeline.sample_sparse_structure(cond_512, 32, 1, {})
            shape_slat, res = pipeline.sample_shape_slat_cascade(
                cond_512, cond_1024,
                models["shape_slat_flow_model_512"], models["shape_slat_flow_model_1024"],
                512, resolution, coords, {}, max_num_tokens,
            )
            tex_slat = pipeline.sample_tex_slat(cond_1024, models["tex_slat_flow_model_1024"], shape_slat, {})
        torch.cuda.empty_cache()
        return pipeline.decode_latent(shape_slat, tex_slat, res)
