"""
Camera sources. Each class exposes the same three methods:

    read()    -> BGR frame as a NumPy array, or None if no frame arrived
    info()    -> short human-readable description
    release() -> close the camera

The rest of the program never needs to know which kind of camera it has.
"""

import time

import cv2

from camera.camera_select import open_capture

# picamera2 only exists on Raspberry Pi OS; everywhere else this is simply skipped.
try:
    from picamera2 import Picamera2
    HAS_PICAMERA2 = True
except ImportError:
    HAS_PICAMERA2 = False


class OpenCVCamera:
    """Laptop webcams and USB cameras."""

    def __init__(self, index, width=None, height=None):
        self.index = index
        self.cap = open_capture(index)
        if self.cap is None:
            raise RuntimeError(f"Could not open camera {index}")
        if width and height:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def read(self):
        ok, frame = self.cap.read()
        return frame if ok else None

    def info(self):
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return f"OpenCV camera {self.index} ({self.cap.getBackendName()}) {w}x{h}"

    def release(self):
        self.cap.release()


class PiCamera:
    """Raspberry Pi camera module (CSI ribbon cable), used on the Pi 5."""

    def __init__(self, width=None, height=None):
        if not HAS_PICAMERA2:
            raise RuntimeError("picamera2 is not installed (it only runs on Raspberry Pi OS)")
        self.size = (width or 640, height or 480)
        self.cam = Picamera2()
        # picamera2's "RGB888" format actually stores pixels as B, G, R in memory,
        # which is exactly the order OpenCV expects, so no color conversion is needed.
        config = self.cam.create_preview_configuration(
            main={"size": self.size, "format": "RGB888"}
        )
        self.cam.configure(config)
        self.cam.start()
        time.sleep(0.5)  # let auto-exposure settle

    def read(self):
        return self.cam.capture_array()

    def info(self):
        return f"Raspberry Pi camera module {self.size[0]}x{self.size[1]}"

    def release(self):
        self.cam.stop()
        self.cam.close()
