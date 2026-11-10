"""
Vision pipeline. Desk-color learning, object detection and tag reading will go here.

The frame -> mask filters live in desk_model.py and filters.py; this module
builds the dashboard's view from them and groups the mask into objects.
"""

import cv2
import numpy as np

from cv.filters import score_to_heatmap

# Filter checkboxes offered by the dashboard. They can be combined freely.
FILTERS = {
    "blur": "Blur",
    "lab_l": "L (lightness)",
    "lab_a": "a (green-red)",
    "lab_b": "b (blue-yellow)",
    "heatmap": "Score heatmap",
    "mask": "Mask",
    "diff": "Highlight differences",
    "grid": "Grid & labels",
}


def process_frame(frame: np.ndarray) -> np.ndarray:
    """Placeholder. Currently returns the frame unchanged."""
    return frame


def apply_filters(frame, filters, mask=None, score=None, blur_ksize=5, limit=4.0):
    """
    Build the displayed picture from the ticked filters, applied in order:
    blur, LAB channels, then heatmap or mask (these replace the picture), then
    the red highlight over the final mask. ("grid" is drawn by the worker,
    since it needs the desk coordinates.)
    Always returns a new 3-channel BGR image the same size as the frame.
    """
    shown = frame
    if filters.get("blur"):
        shown = cv2.GaussianBlur(shown, (blur_ksize, blur_ksize), 0)

    keep = [filters.get("lab_l"), filters.get("lab_a"), filters.get("lab_b")]
    if any(keep):
        # Unticked channels are set to neutral (128), so L alone is grayscale
        # and a + b alone shows only the color information.
        lab = cv2.cvtColor(shown, cv2.COLOR_BGR2LAB)
        for channel, kept in enumerate(keep):
            if not kept:
                lab[:, :, channel] = 128
        shown = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    if filters.get("heatmap") and score is not None:
        shown = score_to_heatmap(score, limit)
    if filters.get("mask") and mask is not None:
        shown = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    shown = shown.copy() if shown is frame else shown
    if filters.get("diff") and mask is not None:
        flagged = mask > 0
        red = np.array([0, 0, 255], dtype=np.float32)
        shown[flagged] = (shown[flagged] * 0.4 + red * 0.6).astype(np.uint8)
    return shown


def find_objects(mask, min_area_px, include_mask=None, exclude_mask=None):
    """
    Group the white pixels of a filters.make_mask mask into objects.

    include_mask: only look inside this area (e.g. the desk between the markers).
    exclude_mask: never count these pixels (e.g. the ArUco markers themselves).
    Returns a list of OpenCV contours, largest first.
    """
    if include_mask is not None or exclude_mask is not None:
        mask = mask.copy()
        if include_mask is not None:
            mask &= include_mask
        if exclude_mask is not None:
            mask[exclude_mask > 0] = 0
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) >= min_area_px]
    return sorted(contours, key=cv2.contourArea, reverse=True)
