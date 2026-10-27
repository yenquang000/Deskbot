"""
The filter pipeline: turn a camera frame into a black-and-white mask where
white means "not desk".

    frame -> blur -> Lab -> score vs desk model -> threshold -> open -> close -> ROI -> mask

Every step takes optional overrides (defaulting to config.py) so the dashboard
can tune them while the camera runs.
"""

from functools import lru_cache

import cv2
import numpy as np

import config
from cv.desk_model import preprocess


@lru_cache(maxsize=16)
def _ellipse(size):
    """An elliptical kernel; cached so each size is only built once, not per frame."""
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


# The default kernels, built once at import time.
OPEN_KERNEL = _ellipse(config.OPEN_KSIZE)
CLOSE_KERNEL = _ellipse(config.CLOSE_KSIZE)


def score_frame(lab, model, weights=config.LAB_WEIGHTS):
    """
    How far each pixel's color is from the desk: a weighted distance in
    standard deviations. Returns float32 (h, w); 0 = exactly desk colored.
    """
    L, a, b = cv2.split(lab)

    # In Lab, darkening a color also pulls a and b toward neutral (128), so a
    # shadow looks slightly less colorful. Scaling a and b by (desk L / pixel L)
    # undoes that: a shadowed desk pixel gets the desk's color back, while a
    # differently colored object stays different.
    ratio = float(model.mean[0]) / np.maximum(L, config.MIN_L_FOR_CHROMA)
    a = cv2.multiply(cv2.subtract(a, 128.0), ratio) + 128.0
    b = cv2.multiply(cv2.subtract(b, 128.0), ratio) + 128.0

    # z = (value - desk mean) / desk spread, weighted per channel. L is weighted
    # low so the remaining brightness change of a shadow scores small.
    total = None
    for channel, mean, std, weight in zip((L, a, b), model.mean, model.std, weights):
        z = cv2.subtract(channel, float(mean)) * (float(weight) / float(std))
        squared = cv2.multiply(z, z)
        total = squared if total is None else cv2.add(total, squared)
    return cv2.sqrt(total)


def score_to_heatmap(score, threshold=config.SCORE_THRESHOLD):
    """Debug view: the score as a colored heatmap (uint8 BGR, black = desk)."""
    top = 3 * threshold
    scaled = (np.clip(score, 0, top) * (255.0 / top)).astype(np.uint8)
    return cv2.applyColorMap(scaled, cv2.COLORMAP_MAGMA)


def threshold(score, limit=config.SCORE_THRESHOLD):
    """uint8 mask: 255 where the score is above the limit ("not desk"), else 0."""
    return np.where(score > limit, 255, 0).astype(np.uint8)


def open_mask(mask, size=config.OPEN_KSIZE):
    """Opening removes isolated speckles smaller than the kernel."""
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, _ellipse(size))


def close_mask(mask, size=config.CLOSE_KSIZE):
    """Closing fills holes and joins nearby pieces of one object (a pen and its cap)."""
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _ellipse(size))


def apply_roi(mask, roi):
    """Zero everything outside roi = (x, y, w, h)."""
    x, y, w, h = roi
    out = np.zeros_like(mask)
    out[y:y + h, x:x + w] = mask[y:y + h, x:x + w]
    return out


def make_mask(frame, model, roi=None, *, blur_ksize=config.BLUR_KSIZE,
              weights=config.LAB_WEIGHTS, limit=config.SCORE_THRESHOLD,
              open_ksize=config.OPEN_KSIZE, close_ksize=config.CLOSE_KSIZE):
    """
    Run the full chain on one frame. roi defaults to the model's learned ROI.
    Returns (final mask, stages) where stages holds every intermediate step for
    debugging: {"lab", "score", "raw", "opened", "closed"}.
    """
    lab = preprocess(frame, blur_ksize)
    score = score_frame(lab, model, weights)
    raw = threshold(score, limit)
    opened = open_mask(raw, open_ksize)
    closed = close_mask(opened, close_ksize)
    mask = apply_roi(closed, model.roi if roi is None else roi)
    return mask, {"lab": lab, "score": score, "raw": raw, "opened": opened, "closed": closed}
