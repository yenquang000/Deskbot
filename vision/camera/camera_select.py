"""
Camera backends and selection: picking the right OpenCV backend for the
operating system, opening a camera index in a way that is confirmed to deliver
frames, finding every working camera, and choosing which one to open.
"""

import platform
import time

import cv2

MAX_INDEX_TO_SCAN = 6  # OpenCV camera indices 0..5


def preferred_backends():
    """The capture backends that work best on each operating system, best first."""
    system = platform.system()
    if system == "Windows":
        return [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
    if system == "Darwin":  # macOS
        return [cv2.CAP_AVFOUNDATION, cv2.CAP_ANY]
    return [cv2.CAP_V4L2, cv2.CAP_ANY]  # Linux, including Raspberry Pi OS


def quiet_opencv(silent=True):
    """OpenCV logs a warning for every empty camera slot it probes; hide that noise."""
    try:
        log = cv2.utils.logging
        level = log.LOG_LEVEL_SILENT if silent else log.LOG_LEVEL_WARNING
        log.setLogLevel(level)
    except AttributeError:
        pass  # very old OpenCV builds lack this; the warnings are harmless


def open_capture(index):
    """
    Try each backend until one opens the camera AND returns a real frame.
    Returns an open cv2.VideoCapture, or None if nothing worked.
    """
    quiet_opencv(True)
    try:
        for backend in preferred_backends():
            cap = cv2.VideoCapture(index, backend)
            if not cap.isOpened():
                cap.release()
                continue
            # Some cameras "open" but never deliver frames; read a few to be sure.
            for _ in range(5):
                ok, frame = cap.read()
                if ok and frame is not None and frame.size > 0:
                    return cap
                time.sleep(0.05)
            cap.release()
        return None
    finally:
        quiet_opencv(False)


def find_cameras():
    """Return the OpenCV indices of every camera that actually produces frames."""
    found = []
    for index in range(MAX_INDEX_TO_SCAN):
        cap = open_capture(index)
        if cap is not None:
            found.append(index)
            cap.release()
    return found


def open_camera(index=None, use_picamera=False, width=None, height=None):
    """
    Open a camera.

    - use_picamera=True  -> the Raspberry Pi camera module
    - index=<n>          -> that specific OpenCV camera
    - neither            -> auto-detect (Pi camera first if available, then 0..5)
    """
    # Imported here, not at the top, because cameras.py imports open_capture
    # from this module; a top-level import would be circular.
    from camera.cameras import HAS_PICAMERA2, OpenCVCamera, PiCamera

    if use_picamera:
        return PiCamera(width, height)

    if index is not None:
        return OpenCVCamera(index, width, height)

    if HAS_PICAMERA2:
        try:
            return PiCamera(width, height)
        except Exception as e:  # no module attached, ribbon loose, etc.
            print(f"[info] Pi camera not available ({e}); trying other cameras.")

    cameras = find_cameras()
    if not cameras:
        raise RuntimeError(
            "No working camera found. Check that it is plugged in, not in use by "
            "another app (Zoom, browser), and that this program has camera permission."
        )
    return OpenCVCamera(cameras[0], width, height)
