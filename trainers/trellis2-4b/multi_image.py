"""Several pictures of one object (front, back, a side…) → one 3D model.

TRELLIS.2 is trained on one picture and its pipeline has no multi-picture entry point. TRELLIS (v1)
shipped a tuning-free one, `run_multi_image`, and it carries over because the samplers are the same
flow-matching Euler samplers: every picture is encoded on its own, and at each sampler step the model's
prediction is steered by the pictures. v1 has two ways to do that:

  in turn    ("stochastic" in v1) one picture per step, in turn, starting with the main picture. Free.
  together   ("multidiffusion" in v1) the mean of one prediction per picture, at every step: one extra
             model call per extra picture per step.

and `together_steps` mixes them: the pictures take turns, then the last `together_steps` steps of every
stage are taken together (0 = v1's stochastic, >= the sampler's steps = v1's multidiffusion).

What about 40 runs on the dev endpoint showed (2026-11-25, five objects, seed 42 — one seed, so read it as
a tendency, not a law):
  - A second picture does what it is for: the back of the model follows the back picture (a backpack's
    shape and colour, a tail, both arms, a hoodie instead of an invented cape) instead of a guess.
  - Two pictures: in turn was clean in 4 of 4. Together, at every step or only at the end, came out
    muddier in 4 of 4 (a grey backpack seen only from behind came out tan or brown).
  - Three or four pictures: in turn leaves each picture few steps, and the picture with the last turn
    repaints the fine detail of the whole object — smudged or wrong texture in 3 of 6. Taking the last
    4 steps together was acceptable in 5 of 5. Together at every step was not reliably better (1 clean,
    1 smudged) and costs up to 4x.
  - More sampler steps (18, 20, 24) did not help, and made one two-picture run worse.
  - Joining the pictures' tokens into one conditioning sequence (not from TRELLIS) put a second face on
    the back of a head. Not kept.
  - Pictures that disagree with each other (AI-drawn "other sides") come out smudged where they disagree,
    whatever the setting.
Hence `default_together_steps`. `together_steps` and `sampler_steps` stay job inputs for tuning.

The mixing sits under classifier-free guidance, so the guidance interval and the CFG rescale of
pipeline.json apply to the mixed prediction exactly as they apply to a single picture's.

`make_inference` and `default_together_steps` are pure Python (no torch), unit-tested on a laptop.
"""
from __future__ import annotations

from contextlib import contextmanager

CASCADE_RESOLUTION = {"1024_cascade": 1024, "1536_cascade": 1536}


def default_together_steps(pictures: int) -> int:
    """Two pictures take turns to the end; three or four take the last 4 steps together (see above)."""
    return 0 if pictures <= 2 else 4


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


def run(pipeline, pictures: list, seed: int, pipeline_type: str, together_steps: int, steps: int | None = None,
        max_num_tokens: int = 49152, timings: dict | None = None) -> list:
    """Trellis2ImageTo3DPipeline.run for the two cascade types, conditioned on several preprocessed
    pictures. Same stages, sampler settings and seeding as upstream, except `steps`: when given, every
    stage's sampler takes that many steps instead of pipeline.json's. `timings` (optional) gets the seconds
    each stage took."""
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
        override = {"steps": steps} if steps else {}
        shape_steps = steps or pipeline.shape_slat_sampler_params["steps"]
        stage_steps = [steps or pipeline.sparse_structure_sampler_params["steps"], shape_steps, shape_steps, steps or pipeline.tex_slat_sampler_params["steps"]]
        with several_pictures(FlowEulerSampler, len(pictures), together_steps, stage_steps):
            coords = pipeline.sample_sparse_structure(cond_512, 32, 1, override)
            lap("structure")
            shape_slat, res = pipeline.sample_shape_slat_cascade(
                cond_512, cond_1024,
                models["shape_slat_flow_model_512"], models["shape_slat_flow_model_1024"],
                512, resolution, coords, override, max_num_tokens,
            )
            lap("shape")
            tex_slat = pipeline.sample_tex_slat(cond_1024, models["tex_slat_flow_model_1024"], shape_slat, override)
            lap("texture")
        torch.cuda.empty_cache()
        meshes = pipeline.decode_latent(shape_slat, tex_slat, res)
        lap("decode")
        return meshes
