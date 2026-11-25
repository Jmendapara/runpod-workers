"""Several pictures of one object (front, back, a side…) → one 3D model.

TRELLIS.2 is trained on one picture and its pipeline has no multi-picture entry point. TRELLIS (v1)
shipped a tuning-free one, `run_multi_image`, and it carries over because the samplers are the same
flow-matching Euler samplers: every picture is encoded on its own, and at each sampler step the model's
prediction is steered by the pictures. v1 has two ways to do that, and each fails in its own way on
TRELLIS.2 (measured on the dev endpoint, 2026-11-25):

  in turn    ("stochastic" in v1) one picture per step, in turn, starting with the main picture. Free.
             But the last steps cross most of the noise range, so whichever picture has the last turn
             repaints the fine detail of the whole object and the other pictures' sides come out smudged
             — worse with every picture added.
  together   ("multidiffusion" in v1) the mean of one prediction per picture, at every step. One extra
             model call per extra picture per step. No picture dominates, but while the noise is still
             high the pictures disagree about colours they cannot see and the mean is mud (a grey
             backpack seen only from behind comes out tan).

So the default does both, in that order: the pictures take turns while the object is being laid out, then
the last `together_steps` steps of every stage are taken together.

  together_steps = 0        v1's stochastic
  together_steps >= steps   v1's multidiffusion

The mixing sits under classifier-free guidance, so the guidance interval and the CFG rescale of
pipeline.json apply to the mixed prediction exactly as they apply to a single picture's.

`make_inference` is pure Python (no torch), so the step logic is unit-tested on a laptop.
"""
from __future__ import annotations

from contextlib import contextmanager

CASCADE_RESOLUTION = {"1024_cascade": 1024, "1536_cascade": 1536}


def make_inference(original, count: int, together_steps: int, stage_steps: list[int]):
    """A replacement for FlowEulerSampler._inference_model. `original` is that method; a conditioning
    whose batch is `count` is the pictures' (the negative conditioning has a batch of one and passes
    through untouched). `stage_steps` is the number of sampler steps of each stage, in the order the
    stages run; every sampler starts at t = 1, which is how a new stage is recognised."""
    state = {"stage": -1, "step": 0}

    def inference(self, model, x_t, t, cond=None, **kwargs):
        if cond is None or cond.shape[0] != count:
            return original(self, model, x_t, t, cond, **kwargs)
        if t >= 1.0 - 1e-6:
            state["stage"] += 1
            state["step"] = 0
        step = state["step"]
        state["step"] += 1
        steps = stage_steps[min(max(state["stage"], 0), len(stage_steps) - 1)]
        if step < steps - together_steps:
            i = step % count  # in turn: each stage begins with the main picture
            return original(self, model, x_t, t, cond[i:i + 1], **kwargs)
        pred = original(self, model, x_t, t, cond[0:1], **kwargs)
        for i in range(1, count):
            pred = pred + original(self, model, x_t, t, cond[i:i + 1], **kwargs)
        return (1.0 / count) * pred

    return inference


@contextmanager
def several_pictures(sampler_cls, count: int, together_steps: int, stage_steps: list[int]):
    """While open, every sampler built on `sampler_cls` mixes `count` pictures. One job runs at a time."""
    original = sampler_cls._inference_model
    sampler_cls._inference_model = make_inference(original, count, together_steps, stage_steps)
    try:
        yield
    finally:
        sampler_cls._inference_model = original


def conditioning(pipeline, pictures: list, resolution: int) -> dict:
    import torch

    cond = pipeline.get_cond(pictures, resolution, include_neg_cond=False)["cond"]  # (pictures, tokens, dim)
    return {"cond": cond, "neg_cond": torch.zeros_like(cond[:1])}


def run(pipeline, pictures: list, seed: int, pipeline_type: str, together_steps: int, max_num_tokens: int = 49152,
        timings: dict | None = None) -> list:
    """Trellis2ImageTo3DPipeline.run for the two cascade types, conditioned on several preprocessed
    pictures. Same stages, sampler settings and seeding as upstream. `timings` (optional) gets the
    seconds each stage took."""
    import time

    import torch
    from trellis2.pipelines.samplers.flow_euler import FlowEulerSampler

    resolution = CASCADE_RESOLUTION[pipeline_type]
    models = pipeline.models
    started = time.time()

    def lap(name: str) -> None:
        nonlocal started
        if timings is not None:
            timings[name] = round(time.time() - started, 2)
        started = time.time()

    with torch.no_grad():
        torch.manual_seed(seed)
        cond_512 = conditioning(pipeline, pictures, 512)
        cond_1024 = conditioning(pipeline, pictures, 1024)
        lap("encode")
        # The stages below, in order: structure, shape at 512, shape at the cascade resolution, texture.
        shape_steps = pipeline.shape_slat_sampler_params["steps"]
        stage_steps = [pipeline.sparse_structure_sampler_params["steps"], shape_steps, shape_steps, pipeline.tex_slat_sampler_params["steps"]]
        with several_pictures(FlowEulerSampler, len(pictures), together_steps, stage_steps):
            coords = pipeline.sample_sparse_structure(cond_512, 32, 1, {})
            lap("structure")
            shape_slat, res = pipeline.sample_shape_slat_cascade(
                cond_512, cond_1024,
                models["shape_slat_flow_model_512"], models["shape_slat_flow_model_1024"],
                512, resolution, coords, {}, max_num_tokens,
            )
            lap("shape")
            tex_slat = pipeline.sample_tex_slat(cond_1024, models["tex_slat_flow_model_1024"], shape_slat, {})
            lap("texture")
        torch.cuda.empty_cache()
        meshes = pipeline.decode_latent(shape_slat, tex_slat, res)
        lap("decode")
        return meshes
