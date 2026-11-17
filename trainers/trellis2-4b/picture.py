"""Clean a picture's transparency before TRELLIS.2 sees it. Pure numpy, unit-tested without a GPU.

TRELLIS.2's preprocessing trusts any alpha channel: a picture with transparency is used as the cut-out
as it is, and only a fully opaque one goes through background removal. Pictures exported from
background-removal tools often keep a faint, invisible residue of the old background (alpha 1–25 over
much of the frame); TRELLIS premultiplies it into a dark haze and builds it as a flat panel around the
subject. So: alpha under ALPHA_FLOOR counts as fully transparent, and a picture that is then (almost)
entirely opaque is handed over without its alpha, for background removal.
"""
from __future__ import annotations

import numpy as np

ALPHA_FLOOR = 64              # under 25 % opacity: not part of the object
MIN_TRANSPARENT_SHARE = 0.01  # less than this transparent: the alpha does not cut anything out


def clean_alpha(rgba: np.ndarray) -> np.ndarray | None:
    """rgba: HxWx4 uint8. Returns the cleaned RGBA array, or None when the picture should be treated as
    opaque (its alpha does not isolate a subject) and go through background removal."""
    alpha = rgba[..., 3]
    cleaned = np.where(alpha < ALPHA_FLOOR, 0, alpha).astype(np.uint8)
    if (cleaned == 0).mean() < MIN_TRANSPARENT_SHARE:
        return None
    out = rgba.copy()
    out[..., 3] = cleaned
    return out
