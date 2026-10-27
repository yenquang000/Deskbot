"""
The live preview window: reads frames, passes them through the vision pipeline,
and displays the result until the user quits. Pressing l learns the desk color.
"""

import cv2

from cv.desk_model import CalibrationError, learn_desk
from cv.vision import process_frame

WINDOW_NAME = "Deskbot Camera"
MAX_FAILED_READS = 30
LEARN_FRAMES = 30  # frames of the clean desk to average over


def capture_frames(camera, count):
    """Read `count` good frames from the camera, keeping the window responsive."""
    frames = []
    failed_reads = 0
    while len(frames) < count:
        frame = camera.read()
        if frame is None:
            failed_reads += 1
            if failed_reads > MAX_FAILED_READS:
                raise RuntimeError("Camera stopped sending frames.")
            continue
        frames.append(frame)
        cv2.imshow(WINDOW_NAME, frame)
        cv2.waitKey(1)
    return frames


def learn_desk_interactive(camera, frame):
    """
    Let the user drag a box around the clean desk, then learn its color.
    Returns (roi, DeskModel), or None if the selection was cancelled or failed.
    """
    print("[info] Drag a box over the clean desk, then press Enter (c = cancel).")
    roi = cv2.selectROI(WINDOW_NAME, frame, showCrosshair=False)
    if roi[2] == 0 or roi[3] == 0:
        print("[info] Desk learning cancelled.")
        return None

    print(f"[info] Capturing {LEARN_FRAMES} frames; keep the desk clear.")
    frames = capture_frames(camera, LEARN_FRAMES)
    try:
        model = learn_desk(frames, roi)
    except CalibrationError as e:
        print(f"[warn] {e}")
        return None
    print(f"[ok] Desk learned. LAB mean={model.mean.round(1)} std={model.std.round(1)}")
    return roi, model


def run(camera):
    """Show the camera feed until q / Esc is pressed or the window is closed."""
    print(f"[ok] Using {camera.info()}")
    print("[ok] Press l to learn the desk color, q or Esc to quit.")
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    desk_roi = None
    desk_model = None  # DeskModel from learn_desk, for detection to use later
    failed_reads = 0
    try:
        while True:
            frame = camera.read()
            if frame is None:
                failed_reads += 1
                if failed_reads > MAX_FAILED_READS:
                    print("[error] Camera stopped sending frames.")
                    break
                continue
            failed_reads = 0

            shown = process_frame(frame)
            if desk_roi is not None:
                x, y, w, h = desk_roi
                cv2.rectangle(shown, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.imshow(WINDOW_NAME, shown)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):  # 27 = Esc
                break
            if key == ord("l"):
                learned = learn_desk_interactive(camera, frame)
                if learned is not None:
                    desk_roi, desk_model = learned
            # Closing the window with the X button also quits.
            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break
    except KeyboardInterrupt:
        pass
    except RuntimeError as e:
        print(f"[error] {e}")
    finally:
        camera.release()
        cv2.destroyAllWindows()
