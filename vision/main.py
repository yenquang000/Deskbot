"""
Deskbot vision - entry point.

This file only reads command-line options and hands off to the other modules.

Usage:
  python main.py              # auto-detect the first working camera, open the dashboard
  python main.py --list       # list every camera that works
  python main.py --camera 1   # use a specific OpenCV camera index
  python main.py --picamera   # force the Raspberry Pi camera module
  python main.py --width 1280 --height 720
  python main.py --host 0.0.0.0   # on the Pi: open the dashboard from another computer
  python main.py --window     # plain OpenCV window instead of the dashboard

Dashboard (default): a browser page with the camera, live data, view options,
settings, and buttons for every command. Keys l, q and Esc work there too.

Controls in the --window viewer:  l = learn desk color,  q or Esc = quit
"""

import argparse
import platform
import sys
from pathlib import Path

import cv2

# Let the vision modules import the shared/ and hardware/ folders from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from camera.camera_select import find_cameras, open_camera  # noqa: E402
from camera.cameras import HAS_PICAMERA2  # noqa: E402
from dashboard.server import run_dashboard  # noqa: E402
from dashboard.viewer import run  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Deskbot camera viewer")
    parser.add_argument("--camera", type=int, help="OpenCV camera index (0, 1, ...)")
    parser.add_argument(
        "--picamera", action="store_true", help="use the Raspberry Pi camera module"
    )
    parser.add_argument("--list", action="store_true", help="list working cameras and exit")
    parser.add_argument("--width", type=int, help="requested frame width")
    parser.add_argument("--height", type=int, help="requested frame height")
    parser.add_argument(
        "--window", action="store_true", help="use the plain OpenCV window, not the dashboard"
    )
    parser.add_argument(
        "--host", default="127.0.0.1",
        help="dashboard address (0.0.0.0 = reachable from other devices)",
    )
    parser.add_argument("--port", type=int, default=8000, help="dashboard port")
    parser.add_argument(
        "--no-browser", action="store_true", help="don't open the dashboard in a browser"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print(f"[info] OpenCV {cv2.__version__} on {platform.system()}")

    if args.list:
        cameras = find_cameras()
        print(f"[info] Working OpenCV cameras: {cameras or 'none'}")
        print(f"[info] picamera2 available: {HAS_PICAMERA2}")
        return 0

    try:
        camera = open_camera(
            index=args.camera,
            use_picamera=args.picamera,
            width=args.width,
            height=args.height,
        )
    except Exception as e:
        print(f"[error] {e}")
        return 1

    if args.window:
        run(camera)
        return 0
    return run_dashboard(camera, args.host, args.port, open_browser=not args.no_browser)


if __name__ == "__main__":
    sys.exit(main())
