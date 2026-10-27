"""
The camera worker: a background thread that reads frames, finds the ArUco
markers, runs the filter pipeline, boxes the flagged objects, draws the chosen
filters, and hands the coordinates to the Pico link. It also keeps the
statistics and event log the dashboard shows. Every public method is safe to
call from the web server's threads.
"""

import math
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from cv.desk_model import CalibrationError, default_roi, learn_desk
from cv.filters import make_mask
from cv.markers import MarkerTracker
from cv.vision import FILTERS, apply_filters, find_objects, process_frame
from shared import config as shared

MAX_FAILED_READS = 30
BLACK_FRAME_SECONDS = 2.0  # warn when the image stays black this long
MARKER_MARGIN = 1.7  # ignore this much around each corner marker (its white paper)
TRACK_DISTANCE_PX = 40  # an object within this distance of last frame's keeps its id
GRID_STEP_MM = 100
VISION_DIR = Path(__file__).resolve().parent.parent
CAPTURES_DIR = VISION_DIR / "captures"

# Every checkbox in the dashboard: the image filters plus two extra overlays.
TOGGLES = {**FILTERS, "fps": "FPS counter", "desk_box": "Learned area"}

GREEN = (0, 220, 0)
YELLOW = (0, 220, 255)
GREY = (150, 150, 150)
WHITE = (255, 255, 255)


def _rounded(values):
    """float32 arrays print as 6.599999904632568; make that 6.6."""
    return [round(float(v), 1) for v in values]


def _text(image, text, origin, color=WHITE, scale=0.5):
    """Readable text: colored letters on a dark box."""
    # A box rather than a thick dark outline: OpenCV draws thicker letters wider,
    # so an outline drifts away from the text along a long label.
    (w, h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    x, y = origin
    cv2.rectangle(image, (x - 2, y - h - 3), (x + w + 2, y + baseline), (0, 0, 0), -1)
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


class CameraWorker(threading.Thread):
    def __init__(self, camera, settings, link=None):
        super().__init__(daemon=True, name="camera-worker")
        self.camera = camera
        self.settings = settings
        self.link = link  # PicoLink, or None
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._new_jpeg = threading.Condition()

        self.jpeg = None
        self.jpeg_id = 0
        self.last_frame = None

        self.paused = False
        self.toggles = {key: False for key in TOGGLES}
        self.toggles.update(fps=True, grid=True)

        self.markers = MarkerTracker(settings.desk_width_mm, settings.desk_height_mm)
        self.desk_frame = None  # latest markers.DeskFrame
        self.desk = None  # desk_model.DeskModel
        self.desk_learned_at = None
        self.learning = None  # {"roi": ..., "frames": [...]} while learning

        self.mask = None
        self.score = None
        self.objects = []  # dicts with pixel and (when calibrated) mm values
        self._next_id = 1

        self.started = time.monotonic()
        self.frames = 0
        self.dropped = 0
        self.failed_in_a_row = 0
        self.fps = 0.0
        self.brightness = 0.0
        self.black_since = None
        self.warning = None
        self.log = deque(maxlen=100)

        self.add_log(f"Using {camera.info()}")

    # ----- event log -------------------------------------------------------

    def add_log(self, message, level="info"):
        with self._lock:
            self.log.append({
                "time": datetime.now().strftime("%H:%M:%S"),
                "level": level,
                "message": message,
            })

    # ----- main loop -------------------------------------------------------

    def run(self):
        last_tick = time.monotonic()
        try:
            while not self._stop_event.is_set():
                frame = self._next_frame()
                if frame is None:
                    continue

                self._update_learning(frame)
                self._update_brightness(frame)
                self._detect(frame)
                self._publish(self._render(frame))

                # Hold to the max FPS setting, and measure the real rate.
                now = time.monotonic()
                spare = 1.0 / self.settings.max_fps - (now - last_tick)
                if spare > 0:
                    self._stop_event.wait(spare)
                    now = time.monotonic()
                elapsed = max(now - last_tick, 1e-6)
                self.fps = 0.9 * self.fps + 0.1 * (1.0 / elapsed) if self.fps else 1.0 / elapsed
                last_tick = now
        except Exception as e:  # keep the dashboard alive and say what went wrong
            self.add_log(f"Camera loop crashed: {e}", "error")
            self.warning = f"Camera loop crashed: {e}"
        finally:
            self.camera.release()
            self.add_log("Camera released.")

    def _next_frame(self):
        """The next frame to show, or None (after a short wait) if none is ready."""
        if self.paused:
            if self.last_frame is None:
                self._stop_event.wait(0.05)
            return self.last_frame

        frame = self.camera.read()
        if frame is None:
            self.dropped += 1
            self.failed_in_a_row += 1
            if self.failed_in_a_row == MAX_FAILED_READS:
                self.warning = "Camera stopped sending frames."
                self.add_log(self.warning, "error")
            self._stop_event.wait(0.01)
            return None

        if self.failed_in_a_row >= MAX_FAILED_READS:
            self.warning = None
            self.add_log("Camera is sending frames again.")
        self.failed_in_a_row = 0
        self.frames += 1
        self.last_frame = process_frame(frame)
        return self.last_frame

    def _update_learning(self, frame):
        if self.learning is None or self.paused:
            return
        self.learning["frames"].append(frame.copy())
        if len(self.learning["frames"]) < self.settings.learn_frames:
            return

        frames, roi = self.learning["frames"], self.learning["roi"]
        with self._lock:
            self.learning = None
        try:
            model = learn_desk(frames, roi, self.settings.std_floor, self.settings.blur_kernel)
        except CalibrationError as e:
            self.add_log(str(e), "warn")
            return
        with self._lock:
            self.desk = model
            self.desk_learned_at = datetime.now().strftime("%H:%M:%S")
        self.add_log(f"Desk learned. Lab mean={_rounded(model.mean)} "
                     f"std={_rounded(model.std)}", "ok")

    def _update_brightness(self, frame):
        self.brightness = float(frame.mean())

        # A covered lens or disabled camera gives frames that are almost all zero.
        if self.brightness < 1.0:
            if self.black_since is None:
                self.black_since = time.monotonic()
            elif (time.monotonic() - self.black_since > BLACK_FRAME_SECONDS
                  and self.warning is None):
                self.warning = "Camera is sending black frames. Is the lens covered?"
                self.add_log(self.warning, "warn")
        elif self.black_since is not None:
            self.black_since = None
            if self.warning and self.warning.startswith("Camera is sending black"):
                self.warning = None
                self.add_log("Camera image is back.")

    # ----- markers, mask and objects ---------------------------------------

    def _detect(self, frame):
        s = self.settings
        info = self.markers.update(frame)
        was_calibrated = self.desk_frame is not None and self.desk_frame.calibrated
        if info.calibrated and not was_calibrated:
            self.add_log("Desk calibrated: all 4 corner markers found.", "ok")
        self.desk_frame = info

        if self.desk is None or self.desk.frame_size != (frame.shape[1], frame.shape[0]):
            self.mask = self.score = None
            self._set_objects([])
            self._send()
            return

        height, width = frame.shape[:2]
        include = None
        roi = None  # the learned area, until the corners give us the whole desk
        if info.calibrated:
            roi = (0, 0, width, height)
            include = np.zeros((height, width), np.uint8)
            cv2.fillPoly(include, [info.desk_polygon_px()], 255)

        self.mask, stages = make_mask(
            frame, self.desk, roi,
            blur_ksize=s.blur_kernel, weights=(s.l_weight, 1.0, 1.0),
            limit=s.diff_threshold, open_ksize=s.open_ksize, close_ksize=s.close_ksize,
        )
        self.score = stages["score"]

        # The corner markers' paper isn't desk colored, so it must not count as
        # an object. The bot's marker is left in: it joins the bot's body into a
        # single blob, which is then recognised as the bot and skipped below.
        exclude = None
        corner_quads = [q for i, q in info.outlines.items() if i in shared.CORNER_IDS]
        if corner_quads:
            exclude = np.zeros((height, width), np.uint8)
            for quad in corner_quads:
                centre = quad.mean(axis=0)
                grown = (quad - centre) * MARKER_MARGIN + centre
                cv2.fillPoly(exclude, [grown.round().astype(np.int32)], 255)

        min_area_px = s.min_object_area
        if info.calibrated:  # the setting is in mm^2 once we know the desk's scale
            desk_px = cv2.contourArea(info.desk_polygon_px().astype(np.float32))
            min_area_px = s.min_object_area * desk_px / (s.desk_width_mm * s.desk_height_mm)

        objects = []
        for contour in find_objects(self.mask, min_area_px, include, exclude):
            if info.bot is not None and cv2.pointPolygonTest(
                    contour, tuple(map(float, info.bot["px"])), False) >= 0:
                continue  # the blob around the bot's marker is the bot itself
            objects.append(self._measure(contour, info))
        self._set_objects(objects)
        self._send()

    @staticmethod
    def _measure(contour, info):
        """Pixel box, plus centre, size and covered area in desk mm when calibrated."""
        x, y, w, h = cv2.boundingRect(contour)
        obj = {"box_px": (x, y, w, h), "area_px": float(cv2.contourArea(contour)),
               "centre_px": (x + w / 2, y + h / 2)}
        if info.calibrated:
            pts = info.to_desk(contour.reshape(-1, 2))
            (xmin, ymin), (xmax, ymax) = pts.min(axis=0), pts.max(axis=0)
            obj.update(x=float((xmin + xmax) / 2), y=float((ymin + ymax) / 2),
                       w=float(xmax - xmin), h=float(ymax - ymin),
                       a=float(cv2.contourArea(pts.astype(np.float32))))
        return obj

    def _set_objects(self, objects):
        """Give each object an id, reusing last frame's id for the nearest match."""
        previous = list(self.objects)
        for obj in objects:
            best, best_dist = None, TRACK_DISTANCE_PX
            for old in previous:
                dist = math.dist(obj["centre_px"], old["centre_px"])
                if dist < best_dist:
                    best, best_dist = old, dist
            if best is not None:
                obj["id"] = best["id"]
                previous.remove(best)
            else:
                obj["id"] = self._next_id
                self._next_id += 1
        with self._lock:
            self.objects = objects

    def _send(self):
        if self.link is None:
            return
        info = self.desk_frame
        calibrated = info is not None and info.calibrated
        mm_objects = [o for o in self.objects if "x" in o]
        self.link.update((self.settings.desk_width_mm, self.settings.desk_height_mm),
                         calibrated, info.bot if calibrated else None, mm_objects)

    # ----- drawing ---------------------------------------------------------

    def _render(self, frame):
        s, t, info = self.settings, self.toggles, self.desk_frame
        shown = apply_filters(frame, t, self.mask, self.score, s.blur_kernel, s.diff_threshold)

        if info is not None:
            if info.calibrated:
                cv2.polylines(shown, [info.desk_polygon_px()], True, GREEN, 1)
                if t["grid"]:
                    self._draw_grid(shown, info)
            for marker_id, (cx, cy) in info.corners_px.items():
                color = GREEN if marker_id in info.visible else GREY
                cv2.circle(shown, (int(cx), int(cy)), 6, color, -1)
                _text(shown, str(marker_id), (int(cx) + 8, int(cy) - 8), color)
            if info.bot is not None:
                self._draw_bot(shown, info.bot, t["grid"])

        for obj in self.objects:
            x, y, w, h = obj["box_px"]
            cv2.rectangle(shown, (x, y), (x + w, y + h), YELLOW, 2)
            label = f"#{obj['id']}"
            if t["grid"] and "x" in obj:
                label += f" {obj['x']:.0f},{obj['y']:.0f}  {obj['w']:.0f}x{obj['h']:.0f}mm"
            _text(shown, label, (x, max(y - 6, 12)), YELLOW)

        if self.learning is not None:
            x, y, rw, rh = self.learning["roi"]
            cv2.rectangle(shown, (x, y), (x + rw, y + rh), YELLOW, 2)
            done = len(self.learning["frames"])
            _text(shown, f"Learning {done}/{s.learn_frames}", (x, max(y - 8, 15)), YELLOW)
        elif t["desk_box"] and self.desk is not None:
            x, y, rw, rh = self.desk.roi
            cv2.rectangle(shown, (x, y), (x + rw, y + rh), GREEN, 1)

        if t["fps"]:
            _text(shown, "PAUSED" if self.paused else f"{self.fps:.0f} fps", (10, 22),
                  GREEN, 0.6)
        return shown

    def _draw_grid(self, image, info):
        w_mm, h_mm = info.size_mm
        lines = []
        for x in np.arange(0, w_mm + 1, GRID_STEP_MM):
            lines.append(((x, 0), (x, h_mm)))
        for y in np.arange(0, h_mm + 1, GRID_STEP_MM):
            lines.append(((0, y), (w_mm, y)))
        overlay = image.copy()
        for start, end in lines:
            a, b = info.to_pixels([start, end]).round().astype(int)
            cv2.line(overlay, tuple(a), tuple(b), WHITE, 1)
        cv2.addWeighted(overlay, 0.35, image, 0.65, 0, dst=image)
        origin = info.to_pixels([(0, 0)])[0].round().astype(int)
        _text(image, "0,0", (origin[0] + 8, origin[1] + 18), GREEN)

    @staticmethod
    def _draw_bot(image, bot, labels):
        centre = tuple(int(v) for v in bot["px"])
        tip = tuple(int(v) for v in bot["tip_px"])
        cv2.polylines(image, [bot["quad_px"].round().astype(np.int32)], True, (255, 160, 0), 2)
        cv2.arrowedLine(image, centre, tip, (255, 160, 0), 3, tipLength=0.3)
        label = "bot"
        if labels:
            label += f" {bot['x']:.0f},{bot['y']:.0f} {bot['hd']:.0f}deg"
        # Below the marker, clear of the heading arrow.
        left, bottom = bot["quad_px"][:, 0].min(), bot["quad_px"][:, 1].max()
        _text(image, label, (int(left), int(bottom) + 16), (255, 160, 0))

    def _publish(self, image):
        ok, buffer = cv2.imencode(
            ".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, self.settings.jpeg_quality]
        )
        if not ok:
            return
        with self._new_jpeg:
            self.jpeg = buffer.tobytes()
            self.jpeg_id += 1
            self._new_jpeg.notify_all()

    # ----- used by the dashboard ------------------------------------------

    def wait_for_jpeg(self, last_id, timeout=1.0):
        """Block until a JPEG newer than `last_id` exists; returns (id, bytes or None)."""
        with self._new_jpeg:
            self._new_jpeg.wait_for(lambda: self.jpeg_id != last_id or self.stopped, timeout)
            return self.jpeg_id, self.jpeg

    @property
    def stopped(self):
        return self._stop_event.is_set()

    def stop(self):
        self._stop_event.set()
        with self._new_jpeg:
            self._new_jpeg.notify_all()

    def start_learning(self, roi=None):
        """
        Begin learning the desk color inside roi = (x, y, w, h), in frame pixels.
        roi=None uses the default centred area (desk_model.default_roi).
        """
        if self.last_frame is None:
            raise ValueError("No camera frame yet")
        if self.paused:
            raise ValueError("Resume the camera before learning the desk")
        fh, fw = self.last_frame.shape[:2]
        if roi is None:
            roi = default_roi(fw, fh)
        x, y, w, h = (int(round(v)) for v in roi)
        x, y = max(0, min(x, fw - 1)), max(0, min(y, fh - 1))
        w, h = min(w, fw - x), min(h, fh - y)
        if w < 2 or h < 2:
            raise ValueError("Desk box is too small")
        with self._lock:
            self.learning = {"roi": (x, y, w, h), "frames": []}
        self.add_log(f"Learning desk in box x={x} y={y} w={w} h={h}; keep the desk clear.")

    def clear_desk(self):
        with self._lock:
            self.desk = self.desk_learned_at = None
            self.learning = None
        self.add_log("Desk model cleared.")

    def recalibrate(self):
        self.markers.reset()
        self.add_log("Corner markers reset; looking for all 4 again.")

    def set_paused(self, paused):
        self.paused = paused
        self.add_log("Paused." if paused else "Resumed.")

    def snapshot(self):
        if self.last_frame is None:
            raise ValueError("No camera frame yet")
        CAPTURES_DIR.mkdir(exist_ok=True)
        path = CAPTURES_DIR / f"snapshot_{datetime.now():%Y%m%d_%H%M%S_%f}.png"
        cv2.imwrite(str(path), self.last_frame)
        self.add_log(f"Snapshot saved to {path.relative_to(VISION_DIR)}", "ok")
        return path.name

    def set_toggles(self, toggles):
        unknown = set(toggles) - set(TOGGLES)
        if unknown:
            raise ValueError(f"Unknown filter: {', '.join(sorted(unknown))}")
        self.toggles = {**self.toggles, **{k: bool(v) for k, v in toggles.items()}}

    def update_settings(self, changes, save=False):
        self.settings = self.settings.validated(changes)
        self.markers.set_desk_size(self.settings.desk_width_mm, self.settings.desk_height_mm)
        if self.link is not None:
            self.link.configure(self.settings.pico_port, self.settings.send_hz)
        if save:
            self.settings.save()
            self.add_log("Settings saved.", "ok")
        else:
            self.add_log("Settings applied.")

    def status(self):
        """Everything the dashboard shows, as JSON-friendly values."""
        with self._lock:
            frame = self.last_frame
            desk = None
            if self.desk is not None:
                desk = {
                    "roi": list(self.desk.roi),
                    "mean": _rounded(self.desk.mean),
                    "std": _rounded(self.desk.std),
                    "learned_at": self.desk_learned_at,
                }
            learning = None
            if self.learning is not None:
                learning = {"done": len(self.learning["frames"]),
                            "total": self.settings.learn_frames}
            objects = list(self.objects)
            log = list(self.log)

        info = self.desk_frame
        calibrated = info is not None and info.calibrated
        bot = None
        if calibrated and info.bot is not None:
            bot = {k: round(info.bot[k]) for k in ("x", "y", "hd")}
        rows = []
        for o in objects:
            if "x" in o:
                rows.append({"id": o["id"], "units": "mm", "x": round(o["x"]),
                             "y": round(o["y"]), "w": round(o["w"]), "h": round(o["h"]),
                             "a": round(o["a"])})
            else:
                x, y, w, h = o["box_px"]
                rows.append({"id": o["id"], "units": "px", "x": round(x + w / 2),
                             "y": round(y + h / 2), "w": w, "h": h, "a": round(o["area_px"])})

        return {
            "camera": self.camera.info(),
            "running": not self.stopped,
            "paused": self.paused,
            "warning": self.warning,
            "stats": {
                "fps": round(self.fps, 1),
                "resolution": f"{frame.shape[1]}x{frame.shape[0]}" if frame is not None else None,
                "frame_size": [frame.shape[1], frame.shape[0]] if frame is not None else None,
                "frames": self.frames,
                "dropped": self.dropped,
                "brightness": round(self.brightness, 1),
                "uptime": int(time.monotonic() - self.started),
            },
            "calibration": {
                "calibrated": calibrated,
                "visible": [] if info is None else info.visible,
                "remembered": [] if info is None else sorted(info.corners_px),
            },
            "bot": bot,
            "objects": rows,
            "desk": desk,
            "learning": learning,
            "toggles": self.toggles,
            "settings": {
                "learn_frames": self.settings.learn_frames,
                "blur_kernel": self.settings.blur_kernel,
                "std_floor": self.settings.std_floor,
                "l_weight": self.settings.l_weight,
                "diff_threshold": self.settings.diff_threshold,
                "open_ksize": self.settings.open_ksize,
                "close_ksize": self.settings.close_ksize,
                "min_object_area": self.settings.min_object_area,
                "desk_width_mm": self.settings.desk_width_mm,
                "desk_height_mm": self.settings.desk_height_mm,
                "send_hz": self.settings.send_hz,
                "pico_port": self.settings.pico_port,
                "jpeg_quality": self.settings.jpeg_quality,
                "max_fps": self.settings.max_fps,
            },
            "pico": None if self.link is None else self.link.status(),
            "log": log,
        }
