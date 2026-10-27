"""
Write printable ArUco marker images for the desk corners and the bot.

Usage (from the vision folder):
  python tools/make_markers.py            # 600-pixel PNGs in vision/markers/
  python tools/make_markers.py --size 800

Print them all at the same size, with a white border around each one. Stick the
corner markers on the desk with their centres at the corners of the area you
want to use, then measure the distance between the centres and enter it in the
dashboard settings (desk width = marker 0 to 1, desk height = marker 0 to 3).
"""

import argparse
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # the repo root
from shared import config as shared  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent.parent / "markers"
NAMES = {
    0: "corner0_origin",
    1: "corner1_plus_x",
    2: "corner2_far",
    3: "corner3_plus_y",
    shared.BOT_ID: "bot",
}


def main():
    parser = argparse.ArgumentParser(description="Write printable ArUco markers")
    parser.add_argument("--size", type=int, default=600, help="marker size in pixels")
    args = parser.parse_args()

    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, shared.ARUCO_DICT))
    OUT_DIR.mkdir(exist_ok=True)
    border = args.size // 6  # the white margin the detector needs around each marker
    for marker_id, name in NAMES.items():
        image = cv2.aruco.generateImageMarker(dictionary, marker_id, args.size)
        image = cv2.copyMakeBorder(image, border, border, border, border,
                                   cv2.BORDER_CONSTANT, value=255)
        cv2.putText(image, f"ID {marker_id}: {name}", (border, border // 2 + 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, 0, 2)
        path = OUT_DIR / f"aruco_{marker_id}_{name}.png"
        cv2.imwrite(str(path), image)
        print(f"[ok] {path}")


if __name__ == "__main__":
    main()
