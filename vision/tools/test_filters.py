"""
Synthetic-image checks for the filter pipeline (config, desk_model, filters).

Usage (from the vision folder):
  python tools/test_filters.py
  python tools/test_filters.py --save-debug debug_out

Prints PASS / FAIL for every check and exits 0 only if all of them pass.
All test images are generated here; no files are needed.
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Import the project modules from the parent (vision) folder, and the repo root
# for the shared settings that config.py reads.
VISION_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(VISION_DIR.parent))
sys.path.insert(0, str(VISION_DIR))

import config  # noqa: E402
from cv.desk_model import CalibrationError, default_roi, learn_desk, preprocess  # noqa: E402
from cv.filters import make_mask, score_frame, score_to_heatmap  # noqa: E402

W, H = 640, 480
DESK_BGR = (150, 190, 215)  # light wood
rng = np.random.default_rng(42)
results = []


# ----- synthetic images ------------------------------------------------------

def add_noise(image, sigma=3.0):
    """Camera sensor noise."""
    noisy = image.astype(np.float32) + rng.normal(0, sigma, image.shape)
    return np.clip(noisy, 0, 255).astype(np.uint8)


def uniform_desk(width=W, height=H):
    """A plain desk-colored image with sensor noise."""
    return add_noise(np.full((height, width, 3), DESK_BGR, np.uint8))


def wood_texture(width, height):
    """Wood grain from wavy sine stripes, plus a gentle left-to-right lighting gradient."""
    y, x = np.mgrid[0:height, 0:width].astype(np.float32)
    grain = np.sin(x * 0.09 + 4 * np.sin(y * 0.013)) + 0.5 * np.sin(x * 0.31 + y * 0.02)
    brightness = 1 + 0.06 * grain  # grain changes brightness by about +-9%
    gradient = 0.92 + 0.16 * (x / width)  # one side of the desk is lit more
    image = np.array(DESK_BGR, np.float32) * (brightness * gradient)[..., None]
    return np.clip(image, 0, 255).astype(np.uint8)


def wood_desk(shift=0):
    """A wood desk frame; `shift` moves the camera view by that many pixels."""
    big = wood_texture(W + 80, H + 80)
    return add_noise(big[20 + shift:20 + shift + H, 20 + shift:20 + shift + W].copy())


def darken(image, region_mask, amount):
    """Darken by `amount` (0.35 = 35% darker) where region_mask is 1, blending smoothly."""
    factor = 1 - amount * region_mask[..., None]
    return np.clip(image.astype(np.float32) * factor, 0, 255).astype(np.uint8)


def soft_shadow_mask(center, radius, softness=41):
    mask = np.zeros((H, W), np.float32)
    cv2.circle(mask, center, radius, 1.0, -1)
    return cv2.GaussianBlur(mask, (softness, softness), 0)


def draw_pen(image):
    """A thick blue pen with a white cap, separated by a hairline gap."""
    cv2.line(image, (220, 260), (400, 300), (170, 60, 20), 12)  # blue body (BGR)
    cv2.line(image, (403, 301), (450, 311), (245, 245, 245), 12)  # white cap


def learned_model(frame):
    return learn_desk([frame], None)


# ----- check helpers ---------------------------------------------------------

def check(name, passed, detail=""):
    results.append(passed)
    print(f"{'PASS' if passed else 'FAIL'}  {name}{f'  ({detail})' if detail else ''}")


def raises(fn, error):
    try:
        fn()
    except error:
        return True
    except Exception:
        return False
    return False


def mask_pixels(frame, model):
    mask, _ = make_mask(frame, model)
    return int(np.count_nonzero(mask))


# ----- phases ----------------------------------------------------------------

def phase0():
    names = ["BLUR_KSIZE", "LAB_WEIGHTS", "STD_FLOOR", "MAX_L_STD", "SCORE_THRESHOLD",
             "OPEN_KSIZE", "CLOSE_KSIZE", "ROI_FRACTION"]
    check("config has every constant", all(hasattr(config, n) for n in names))


def phase1():
    model = learned_model(uniform_desk())
    floored = np.allclose(model.std[1:], config.STD_FLOOR[1:])
    check("uniform desk: a and b std equal the floor", floored, f"std={model.std.round(2)}")

    half = uniform_desk()
    half[:, W // 2:] = 0
    check("half desk / half black raises CalibrationError",
          raises(lambda: learn_desk([half], (0, 0, W, H)), CalibrationError))

    check("frames of different sizes raise ValueError",
          raises(lambda: learn_desk([uniform_desk(), uniform_desk(320, 240)]), ValueError))


def phase2():
    desk = uniform_desk()
    model = learned_model(desk)
    score = score_frame(preprocess(uniform_desk()), model)
    p99 = float(np.percentile(score, 99))
    check("clean desk: 99th percentile score below threshold", p99 < config.SCORE_THRESHOLD,
          f"p99={p99:.2f}")

    shadow_region = np.zeros((H, W), np.float32)
    shadow_region[150:330, 200:440] = 1
    shaded = darken(uniform_desk(), shadow_region, 0.35)  # 65% brightness
    score = score_frame(preprocess(shaded), model)
    med = float(np.median(score[170:310, 220:420]))
    check("shadow (65% brightness): median score below threshold",
          med < config.SCORE_THRESHOLD, f"median={med:.2f}")

    blue = uniform_desk()
    cv2.rectangle(blue, (250, 180), (390, 300), (200, 80, 20), -1)
    score = score_frame(preprocess(blue), model)
    med = float(np.median(score[200:280, 270:370]))
    check("blue rectangle: median score above 3x threshold",
          med > 3 * config.SCORE_THRESHOLD, f"median={med:.1f}")

    heat = score_to_heatmap(score)
    check("score_to_heatmap returns uint8 (H, W, 3)",
          heat.dtype == np.uint8 and heat.shape == (H, W, 3))


def phase3():
    model = learned_model(wood_desk())

    n = mask_pixels(wood_desk(), model)
    check("clean desk: final mask all zeros", n == 0, f"{n} white pixels")

    noisy = wood_desk()
    count = int(0.01 * W * H)
    ys, xs = rng.integers(0, H, count), rng.integers(0, W, count)
    noisy[ys, xs] = rng.integers(0, 256, (count, 3))
    n = mask_pixels(noisy, model)
    check("1% salt-and-pepper noise: final mask all zeros", n == 0, f"{n} white pixels")

    pen = wood_desk()
    draw_pen(pen)
    mask, _ = make_mask(pen, model)
    components = cv2.connectedComponents(mask)[0] - 1  # minus the background
    check("pen with cap: exactly one connected component", components == 1,
          f"{components} components")

    crumbs = wood_desk()
    spots = [(200, 150), (420, 200), (320, 330)]
    for spot in spots:
        cv2.circle(crumbs, spot, 3, (40, 60, 90), -1)
    mask, _ = make_mask(crumbs, model)
    found = sum(bool(mask[y - 2:y + 3, x - 2:x + 3].any()) for x, y in spots)
    check("small crumbs (r=3) survive the noise filtering", found == len(spots),
          f"{found}/{len(spots)} found")

    outside = wood_desk()
    x, y, w, h = default_roi(W, H)
    cv2.rectangle(outside, (5, 5), (x - 10, y - 10), (200, 80, 20), -1)
    n = mask_pixels(outside, model)
    check("object outside the ROI is not in the mask", n == 0, f"{n} white pixels")

    shaded = darken(wood_desk(), soft_shadow_mask((330, 240), 120), 0.35)
    n = mask_pixels(shaded, model)
    check("soft shadow (35%): final mask all zeros", n == 0, f"{n} white pixels")

    n = mask_pixels(wood_desk(shift=35), model)
    check("camera shifted 35 px, no objects: final mask all zeros", n == 0,
          f"{n} white pixels")


# ----- debug grid and timing -------------------------------------------------

def example_scene():
    """Crumbs, a paper clip, a paper scrap, a pen and a shadow on the wood desk."""
    scene = wood_desk()
    for cx, cy in [(180, 150), (196, 162), (470, 140), (300, 380), (330, 372)]:
        cv2.circle(scene, (cx, cy), 3, (40, 60, 90), -1)  # crumbs
    cv2.ellipse(scene, (480, 330), (34, 12), 15, 0, 360, (170, 170, 175), 2)  # paper clip
    cv2.ellipse(scene, (484, 331), (24, 7), 15, 0, 360, (170, 170, 175), 2)
    paper = np.array([[120, 280], [205, 265], [215, 340], [130, 350]], np.int32)
    cv2.fillPoly(scene, [paper], (245, 248, 250))  # paper scrap
    draw_pen(scene)
    return darken(scene, soft_shadow_mask((520, 200), 90), 0.35)


def save_debug(folder):
    model = learned_model(wood_desk())
    frame = example_scene()
    mask, stages = make_mask(frame, model)
    tiles = [
        ("frame", frame),
        ("score heatmap", score_to_heatmap(stages["score"])),
        ("raw threshold", stages["raw"]),
        ("opened", stages["opened"]),
        ("closed", stages["closed"]),
        ("final mask", mask),
    ]
    images = []
    for label, image in tiles:
        tile = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image.copy()
        cv2.putText(tile, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 5)
        cv2.putText(tile, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        images.append(tile)
    grid = np.vstack([np.hstack(images[:3]), np.hstack(images[3:])])
    out = Path(folder)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "filter_stages.png"
    cv2.imwrite(str(path), grid)
    print(f"[debug] wrote {path}")


def time_make_mask(runs=50):
    model = learned_model(wood_desk())
    frame = example_scene()
    make_mask(frame, model)  # warm up
    start = time.perf_counter()
    for _ in range(runs):
        make_mask(frame, model)
    return (time.perf_counter() - start) / runs * 1000


def main():
    parser = argparse.ArgumentParser(description="Check the filter pipeline")
    parser.add_argument("--save-debug", metavar="FOLDER", help="write a stage grid PNG here")
    args = parser.parse_args()

    for phase in (phase0, phase1, phase2, phase3):
        phase()
    print(f"\n{sum(results)}/{len(results)} checks passed")
    print(f"make_mask: {time_make_mask():.1f} ms/frame at {W}x{H}")
    if args.save_debug:
        save_debug(args.save_debug)
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
