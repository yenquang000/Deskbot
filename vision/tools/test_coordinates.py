"""
Synthetic checks for the desk coordinate system, object boxes and the Pico link.

Builds a top-down desk (1 px = 1 mm) with the 4 corner markers, the bot marker
and objects of known size, then warps it to a tilted camera view and checks
the measured millimetre values.

Usage (from the vision folder):
  python tools/test_coordinates.py
  python tools/test_coordinates.py --save-debug debug_out
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np

VISION_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(VISION_DIR.parent))
sys.path.insert(0, str(VISION_DIR))

import config  # noqa: E402
from hardware import messaging, pico_controller  # noqa: E402
from shared import config as shared  # noqa: E402
from dashboard.worker import CameraWorker  # noqa: E402

DESK_W, DESK_H = 1000, 600  # mm, marker centre to centre
MARGIN = 120  # mm of desk visible outside the markers
CAM_W, CAM_H = 640, 480
CORNER_MM = 80  # printed corner marker size
BOT_MM = 70
DESK_BGR = (150, 190, 215)
# Where the four desk corners land in the camera image: a tilted view.
CAMERA_QUAD = np.float32([[95, 75], [555, 62], [600, 420], [55, 405]])
DICTIONARY = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, shared.ARUCO_DICT))
rng = np.random.default_rng(7)
results = []


def check(name, passed, detail=""):
    results.append(passed)
    print(f"{'PASS' if passed else 'FAIL'}  {name}{f'  ({detail})' if detail else ''}")


# ----- scene building --------------------------------------------------------

def paste_marker(canvas, marker_id, centre_mm, size_mm, heading_deg):
    """Draw a marker (with its white border) whose top edge faces heading_deg."""
    pixels = 200
    image = cv2.aruco.generateImageMarker(DICTIONARY, marker_id, pixels)
    border = pixels // 6
    image = cv2.copyMakeBorder(image, border, border, border, border,
                               cv2.BORDER_CONSTANT, value=255)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    total = image.shape[0]

    # Forward (top edge direction) and right vectors in desk mm (y points down).
    t = math.radians(heading_deg)
    f = np.array([math.cos(t), math.sin(t)])
    r = np.array([-f[1], f[0]])
    s = size_mm * total / pixels / 2  # half the size including the border
    c = np.array(centre_mm) + MARGIN
    dst = np.float32([c + f * s - r * s, c + f * s + r * s, c - f * s + r * s, c - f * s - r * s])
    src = np.float32([[0, 0], [total, 0], [total, total], [0, total]])
    warp = cv2.getPerspectiveTransform(src, dst)
    size = (canvas.shape[1], canvas.shape[0])
    warped = cv2.warpPerspective(image, warp, size)
    covered = cv2.warpPerspective(np.full(image.shape[:2], 255, np.uint8), warp, size)
    canvas[covered > 0] = warped[covered > 0]


def top_down(objects=True, bot=True, hide_corner=None):
    """The desk seen from straight above, 1 px per mm, with a margin around it."""
    h, w = DESK_H + 2 * MARGIN, DESK_W + 2 * MARGIN
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    grain = 1 + 0.05 * np.sin(x * 0.05 + 3 * np.sin(y * 0.01))
    canvas = np.clip(np.array(DESK_BGR, np.float32) * grain[..., None], 0, 255).astype(np.uint8)

    corners = [(0, 0), (DESK_W, 0), (DESK_W, DESK_H), (0, DESK_H)]
    for marker_id, centre in zip(shared.CORNER_IDS, corners):
        if marker_id != hide_corner:
            paste_marker(canvas, marker_id, centre, CORNER_MM, 270)  # upright

    if objects:  # a red box 80 x 60 mm centred at (650, 150), and a blue one
        cv2.rectangle(canvas, (650 - 40 + MARGIN, 150 - 30 + MARGIN),
                      (650 + 40 + MARGIN, 150 + 30 + MARGIN), (30, 30, 210), -1)
        cv2.rectangle(canvas, (200 + MARGIN, 420 + MARGIN), (320 + MARGIN, 480 + MARGIN),
                      (190, 90, 30), -1)
    if bot:  # a dark round bot at (400, 300) facing 30 degrees
        cv2.circle(canvas, (400 + MARGIN, 300 + MARGIN), 75, (60, 60, 60), -1)
        paste_marker(canvas, shared.BOT_ID, (400, 300), BOT_MM, 30)
    return canvas


def camera_view(canvas):
    """Warp the top-down desk into the tilted camera image, with sensor noise."""
    src = np.float32([[MARGIN, MARGIN], [MARGIN + DESK_W, MARGIN],
                      [MARGIN + DESK_W, MARGIN + DESK_H], [MARGIN, MARGIN + DESK_H]])
    # Warp at twice the size, then shrink with area averaging, like a real lens
    # does. A direct warp skips pixels when shrinking, which garbles the small
    # marker patterns in a way a real camera never would.
    warp = cv2.getPerspectiveTransform(src, CAMERA_QUAD * 2)
    big = cv2.warpPerspective(canvas, warp, (CAM_W * 2, CAM_H * 2),
                              borderMode=cv2.BORDER_REPLICATE)
    frame = cv2.resize(big, (CAM_W, CAM_H), interpolation=cv2.INTER_AREA)
    noisy = frame.astype(np.float32) + rng.normal(0, 2.5, frame.shape)
    return np.clip(noisy, 0, 255).astype(np.uint8)


class SceneCamera:
    """A fake camera that always shows the given frame."""

    def __init__(self, frame):
        self.frame = frame

    def read(self):
        return self.frame.copy()

    def info(self):
        return "Synthetic desk scene 640x480"

    def release(self):
        pass


# ----- checks ----------------------------------------------------------------

def learned_worker():
    """A worker that has learned the clean desk (no objects, no bot)."""
    clean = camera_view(top_down(objects=False, bot=False))
    worker = CameraWorker(SceneCamera(clean), config.Settings())
    worker.settings = worker.settings.validated({"learn_frames": 1})
    worker.last_frame = clean
    worker.start_learning(None)
    worker._update_learning(clean)
    return worker


def check_coordinates():
    worker = learned_worker()
    check("desk learned from the clean scene", worker.desk is not None)

    scene = camera_view(top_down())
    worker._detect(scene)
    info = worker.desk_frame
    check("all 4 corner markers found", info.calibrated and len(info.visible) == 4,
          f"visible={info.visible}")

    bot = info.bot
    if bot is None:
        check("bot marker found", False)
    else:
        err = math.dist((bot["x"], bot["y"]), (400, 300))
        check("bot position within 10 mm", err < 10, f"({bot['x']:.0f}, {bot['y']:.0f})")
        turn = (bot["hd"] - 30 + 180) % 360 - 180
        check("bot heading within 5 degrees", abs(turn) < 5, f"{bot['hd']:.1f} deg")

    mm_objects = [o for o in worker.objects if "x" in o]
    check("two objects found, the bot is not one of them", len(mm_objects) == 2,
          f"{len(mm_objects)} objects")
    red = min(mm_objects, key=lambda o: math.dist((o["x"], o["y"]), (650, 150)),
              default=None)
    if red is not None:
        err = math.dist((red["x"], red["y"]), (650, 150))
        check("red box centre within 10 mm", err < 10, f"({red['x']:.0f}, {red['y']:.0f})")
        size_ok = abs(red["w"] - 80) < 10 and abs(red["h"] - 60) < 10
        check("red box size within 10 mm", size_ok, f"{red['w']:.0f} x {red['h']:.0f} mm")
        area_ok = abs(red["a"] - 4800) < 0.2 * 4800
        check("red box area within 20%", area_ok, f"{red['a']:.0f} mm^2, expected 4800")

    ids = sorted(o["id"] for o in worker.objects)
    worker._detect(camera_view(top_down()))
    check("object ids stay the same between frames",
          sorted(o["id"] for o in worker.objects) == ids, f"{ids}")

    worker._detect(camera_view(top_down(hide_corner=2)))
    check("calibration kept when one corner is covered", worker.desk_frame.calibrated,
          f"visible={worker.desk_frame.visible}")

    worker.recalibrate()
    worker._detect(camera_view(top_down(hide_corner=2)))
    check("recalibrate forgets the covered corner", not worker.desk_frame.calibrated)
    return worker


def check_messages():
    bot = {"x": 412.4, "y": 230.6, "hd": 359.7}
    objs = [{"id": 3, "x": 610.2, "y": 145.0, "w": 80.4, "h": 60.0, "a": 4210.6}]
    line = messaging.encode(17, (1000, 600), True, bot, objs)
    decoded = pico_controller.parse(line.decode())
    expected = {"seq": 17, "desk": [1000, 600], "cal": True,
                "bot": {"x": 412, "y": 231, "hd": 0},
                "objs": [{"id": 3, "x": 610, "y": 145, "w": 80, "h": 60, "a": 4211}]}
    same = decoded is not None and all(decoded[k] == v for k, v in expected.items())
    check("encode -> Pico parse round trip", same, line.decode().strip())

    uncal = pico_controller.parse(messaging.encode(1, (1000, 600), False, bot, objs).decode())
    check("uncalibrated message carries no coordinates",
          uncal["bot"] is None and uncal["objs"] == [])
    check("Pico rejects a garbled line", pico_controller.parse('{"seq": 1, "cal"') is None)

    # A loopback serial port echoes what we send, standing in for the Pico.
    link = messaging.PicoLink("loop://", send_hz=20)
    link.update((1000, 600), True, bot, objs)
    link.start()
    time.sleep(1.0)
    link.stop()
    link.join(2)
    status = link.status()
    check("PicoLink sends at about the set rate", 15 <= status["sent"] <= 23,
          f"{status['sent']} messages in 1 s at 20/s")
    echoed = status["last_reply"] and json.loads(status["last_reply"])["seq"] > 0
    check("PicoLink reads lines coming back", bool(echoed))

    missing = messaging.PicoLink("/dev/ttyNOPE0", send_hz=10)
    missing.start()
    time.sleep(0.3)
    missing.stop()
    missing.join(2)
    s = missing.status()
    check("missing Pico reports 'not connected'", not s["connected"] and bool(s["error"]),
          s["error"])


def save_debug(worker, folder):
    worker.toggles.update(diff=True, grid=True)
    worker.recalibrate()
    scene = camera_view(top_down())
    worker._detect(scene)
    out = Path(folder)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "coordinates.png"
    cv2.imwrite(str(path), worker._render(scene))
    print(f"[debug] wrote {path}")


def main():
    parser = argparse.ArgumentParser(description="Check desk coordinates and the Pico link")
    parser.add_argument("--save-debug", metavar="FOLDER", help="write an annotated frame here")
    args = parser.parse_args()

    worker = check_coordinates()
    check_messages()
    print(f"\n{sum(results)}/{len(results)} checks passed")
    if args.save_debug:
        save_debug(worker, args.save_debug)
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
