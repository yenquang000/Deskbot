"""
Preprocessing and the desk model: learning what the clean desk's color looks like.

The camera can move, so we never compare a pixel with the same pixel in a
stored photo. Instead we learn the desk's color statistics once (a typical
color and how much it varies) and later score every pixel against them.
"""

from dataclasses import dataclass

import cv2
import numpy as np

import config


@dataclass
class DeskModel:
    mean: np.ndarray  # shape (3,), float32 - median L, a, b of desk pixels
    std: np.ndarray  # shape (3,), float32 - robust spread, already floored
    roi: tuple  # (x, y, w, h) in pixels
    frame_size: tuple  # (width, height) at learning time


class CalibrationError(Exception):
    """The frames given to learn_desk don't look like a single clean desk."""


def preprocess(frame, blur_ksize=config.BLUR_KSIZE):
    """
    Median filter, blur, then convert BGR to Lab as float32.
    Learning and filtering must both call this exact function so they match.
    """
    # The median filter deletes isolated odd pixels (salt-and-pepper noise)
    # outright, while anything a few pixels wide, like a crumb, passes through.
    # A Gaussian blur alone would only smear such pixels into small blobs.
    filtered = cv2.medianBlur(frame, config.MEDIAN_KSIZE)
    # Blurring then averages away ordinary sensor noise.
    blurred = cv2.GaussianBlur(filtered, (blur_ksize, blur_ksize), 0)
    # Lab separates brightness (L) from color (a, b), which lets us care less
    # about shadows (brightness changes) than about real color changes.
    return cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB).astype(np.float32)


def default_roi(width, height):
    """A centered box covering ROI_FRACTION of the width and height, as (x, y, w, h)."""
    w = int(width * config.ROI_FRACTION)
    h = int(height * config.ROI_FRACTION)
    return ((width - w) // 2, (height - h) // 2, w, h)


def learn_desk(frames, roi=None, std_floor=config.STD_FLOOR, blur_ksize=config.BLUR_KSIZE):
    """
    Learn the clean desk's color from a list of frames (BGR uint8, all one size).
    roi: (x, y, w, h) of clean desk to learn from; defaults to default_roi.
    Raises ValueError for mixed frame sizes, CalibrationError for a bad scene.
    """
    if not frames:
        raise ValueError("learn_desk needs at least one frame")
    height, width = frames[0].shape[:2]
    if any(f.shape[:2] != (height, width) for f in frames):
        raise ValueError("All frames must be the same size")
    if roi is None:
        roi = default_roi(width, height)

    x, y, w, h = roi
    pixels = np.concatenate(
        [preprocess(f, blur_ksize)[y:y + h, x:x + w].reshape(-1, 3) for f in frames]
    )
    # Median and percentiles instead of mean and std: a few odd pixels (a crumb,
    # a glint) barely move them.
    mean = np.median(pixels, axis=0).astype(np.float32)
    p10, p90 = np.percentile(pixels, [10, 90], axis=0)
    std = (p90 - p10) / 2.56  # for normal data, the 10-90% range is about 2.56 std
    # A floor stops a very uniform desk from making the detector hypersensitive.
    std = np.maximum(std, std_floor).astype(np.float32)

    if std[0] > config.MAX_L_STD:
        raise CalibrationError(
            "The learned area's brightness varies too much "
            f"(std L={std[0]:.1f}, a={std[1]:.1f}, b={std[2]:.1f}; "
            f"max L is {config.MAX_L_STD:.0f}). Clear the desk and keep the desk edge "
            "out of the box, then try again."
        )
    return DeskModel(mean=mean, std=std, roi=tuple(int(v) for v in roi),
                     frame_size=(width, height))
