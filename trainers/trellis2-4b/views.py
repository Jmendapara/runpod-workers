"""Which way to show a generated model: the turntable starts, and the app's 3D viewer opens, from the
view that looks most like the picture the model was made from.

TRELLIS.2 does not put an object's front in a fixed direction — a character may come out facing +Y or
-X — so a fixed start angle shows a side or the back about half the time. The handler renders the model
from YAW_STEPS directions around it (small, fast) and keeps the one whose silhouette (and, to tell a
front from a mirrored back, colours) best match the cut-out input. Pure numpy, so it is unit-tested
without a GPU.
"""
from __future__ import annotations

import math

import numpy as np

YAW_STEPS = 24        # one candidate every 15°
MATCH_SIZE = 96       # silhouettes are compared at this size
COLOR_WEIGHT = 0.5    # how much colour agreement counts next to the silhouette IoU


def candidate_yaws(steps: int = YAW_STEPS) -> list[float]:
    return [2 * math.pi * i / steps for i in range(steps)]


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    try:
        from scipy.ndimage import binary_fill_holes
        return binary_fill_holes(mask)
    except Exception:  # scipy is in the image; keep the module importable without it
        return mask


def normalize_view(rgb: np.ndarray, mask: np.ndarray, size: int = MATCH_SIZE) -> tuple[np.ndarray, np.ndarray] | None:
    """Crop to the object, centre it in a square (as the pipeline's preprocessing does), resize.
    rgb: HxWx3 float 0..1, mask: HxW bool. Returns (rgb, mask) at size x size, or None when empty."""
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return None
    cy, cx = (ys.min() + ys.max()) / 2, (xs.min() + xs.max()) / 2
    side = max(ys.max() - ys.min(), xs.max() - xs.min()) + 1
    # Sample the square window by nearest neighbour (no PIL needed; the window may leave the frame).
    grid = (np.arange(size) + 0.5) / size * side - side / 2
    yy = np.clip(np.round(cy + grid).astype(int), -1, mask.shape[0])
    xx = np.clip(np.round(cx + grid).astype(int), -1, mask.shape[1])
    inside = ((yy >= 0) & (yy < mask.shape[0]))[:, None] & ((xx >= 0) & (xx < mask.shape[1]))[None, :]
    yc, xc = np.clip(yy, 0, mask.shape[0] - 1), np.clip(xx, 0, mask.shape[1] - 1)
    m = mask[yc][:, xc] & inside
    c = rgb[yc][:, xc] * m[..., None]
    return c, m


def target_from_cutout(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """The pipeline's preprocessed picture: RGB premultiplied by alpha on black. The silhouette is
    everything not black, with enclosed holes (dark parts of the object itself) filled."""
    mask = _fill_holes(rgb.max(axis=2) > 0.03)
    return normalize_view(rgb, mask)


def score_view(target: tuple[np.ndarray, np.ndarray], view: tuple[np.ndarray, np.ndarray] | None) -> float:
    if view is None:
        return -1.0
    trgb, tmask = target
    vrgb, vmask = view
    union = (tmask | vmask).sum()
    if union == 0:
        return -1.0
    both = tmask & vmask
    iou = both.sum() / union
    color = float(np.abs(trgb - vrgb)[both].mean()) if both.any() else 1.0
    return float(iou - COLOR_WEIGHT * color)


def best_yaw(target: tuple[np.ndarray, np.ndarray], views: list[tuple[np.ndarray, np.ndarray] | None], yaws: list[float]) -> tuple[float, float]:
    """(yaw, score) of the best-matching view."""
    scores = [score_view(target, v) for v in views]
    i = int(np.argmax(scores))
    return yaws[i], scores[i]


def viewer_orbit(yaw: float, pitch: float) -> tuple[float, float]:
    """A TRELLIS camera (yaw about +Z from +Y toward +X, pitch above the horizon) as model-viewer's
    camera-orbit (theta from +Z toward +X, phi from +Y) in the exported GLB, whose axes are TRELLIS
    (x, y, z) → glTF (x, z, -y). Degrees."""
    theta = (180.0 - math.degrees(yaw)) % 360.0
    phi = 90.0 - math.degrees(pitch)
    return theta, phi
