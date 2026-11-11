"""RobotHardware backed by the vision program: the camera gives the pose and the
objects, the Pico carries out the moves. Needs only NumPy.

`link` is the PicoLink from hardware/messaging.py (or anything with the same
latest / send_command / wait_done methods, like the fake one in test_vision_bridge.py).
Vision works in mm and degrees; everything returned here is in the sim's cm and radians.
The axes are the same: heading 0 = +x, positive rotation goes from +x toward +y.
"""
import time

import numpy as np

from env import Measurements
from obs import travel_room, wrap
from pi_runner import RobotHardware

# An object is a pen if it is long and thin. The box is axis-aligned, so a diagonal
# pen has a nearly square box; it is caught by how little of the box it fills.
PEN_MIN_LONG_MM = 80.0    # long side of the box must be at least this
PEN_MIN_RATIO = 3.0       # long side / short side
PEN_MAX_FILL = 0.35       # covered area / box area
PEN_DELIVERED_CM = 15.0   # a pen this close to the pen corner is done (as in env.step)

MAX_POSE_AGE_S = 0.5      # older vision data than this is not trusted
CHECK_STEP_TOL = 0.20     # movement check: step length may be off by this fraction
CHECK_TURN_TOL_DEG = 10.0
CHECK_MARGIN_CM = 8.0     # same as the EdgeShield margin


def is_pen(obj):
    """True if a vision object {w, h, a in mm} looks like a pen."""
    long_side, short_side = max(obj["w"], obj["h"]), max(min(obj["w"], obj["h"]), 1e-6)
    thin = long_side / short_side >= PEN_MIN_RATIO
    sparse = obj["a"] / (long_side * short_side) < PEN_MAX_FILL
    return long_side >= PEN_MIN_LONG_MM and (thin or sparse)


class VisionHardware(RobotHardware):

    def __init__(self, link, m=None, log=print, move_timeout=5.0, settle=0.3):
        self.link = link
        self.m = m or Measurements()
        self.log = log
        self.move_timeout = move_timeout
        self.settle = settle  # seconds to let the camera catch up after a move
        self._desk_checked = False

    # ----- what the camera sees ----------------------------------------------
    def _state(self):
        """Fresh, calibrated vision data with the bot visible, or None."""
        state, age = self.link.latest()
        if state is None or age > MAX_POSE_AGE_S:
            return None
        desk_size, calibrated, bot, objects = state
        if not calibrated or bot is None:
            return None
        return state

    def pose(self):
        """((x, y) in cm, heading in rad) from the camera, or None if it can't be trusted."""
        state = self._state()
        if state is None:
            return None
        bot = state[2]
        return np.array([bot["x"], bot["y"]]) / 10.0, float(wrap(np.deg2rad(bot["hd"])))

    def desk_cm(self):
        """(W, H) in cm: the distance between corner marker centres. None before any data."""
        state, _ = self.link.latest()
        return None if state is None else (state[0][0] / 10.0, state[0][1] / 10.0)

    def wall_room_cm(self):
        pos, theta = self.pose()
        W, H = self.desk_cm()
        return [travel_room(pos, theta + off, W, H, self.m.robot_radius)
                for off in (0.0, np.pi, np.pi / 2, -np.pi / 2)]

    def detections(self):
        state = self._state()
        if state is None:
            return None, None
        desk, _, bot, objects = state
        pos = np.array([bot["x"], bot["y"]]) / 10.0
        corner = np.array([desk[0], desk[1]]) / 10.0 - self.m.pen_corner_inset
        debris, pens = [], []
        for o in objects:
            p = np.array([o["x"], o["y"]]) / 10.0
            if not is_pen(o):
                debris.append(p)
            elif np.linalg.norm(p - corner) >= PEN_DELIVERED_CM:
                pens.append(p)
        near = lambda q: np.linalg.norm(q - pos)  # noqa: E731
        return min(debris, key=near, default=None), min(pens, key=near, default=None)

    # ----- what the Pico does ------------------------------------------------
    def _command(self, do, a, timeout):
        if not self.link.wait_done(self.link.send_command(do, a), timeout):
            raise TimeoutError(f"no DONE from the Pico for '{do}' within {timeout:.1f} s "
                               "(is it connected and running pico_controller.py?)")

    def execute(self, action):
        self._command("move", int(action), self.move_timeout)
        time.sleep(self.settle)

    def clean(self, ms=1500):
        self._command("clean", int(ms), ms / 1000.0 + self.move_timeout)

    def stop(self):
        try:
            self.link.wait_done(self.link.send_command("stop"), 1.0)
        except Exception:  # the link may already be down; stopping must never fail
            pass

    # ----- start-up ----------------------------------------------------------
    def wait_ready(self, should_stop=None, timeout=None):
        """Block until the desk is calibrated and the bot is visible. False if
        should_stop() turns True or `timeout` seconds pass first."""
        deadline = None if timeout is None else time.monotonic() + timeout
        while self.pose() is None:
            if should_stop is not None and should_stop():
                return False
            if deadline is not None and time.monotonic() > deadline:
                return False
            time.sleep(0.1)
        if not self._desk_checked:
            self._desk_checked = True
            warning = self.desk_warning()
            if warning:
                self.log(warning)
        return True

    def desk_warning(self):
        """A message if the vision desk is outside the sizes the policy trained on, else None."""
        W, H = self.desk_cm()
        m = self.m
        if max(abs(W / m.desk_width - 1), abs(H / m.desk_height - 1)) <= 0.05:
            return None
        return (f"WARNING: the vision desk is {W:.1f} x {H:.1f} cm but the policy trained on "
                f"{m.desk_width} x {m.desk_height} cm (+-5%). It has not seen this size: match "
                "the marker spacing to the training desk, or set Measurements and retrain.")


def movement_check(hw, log=print, should_stop=None, pause=0.5):
    """Run each action once and compare the camera-measured move with the expected one
    (base_step along the heading, +-turn_deg). Forward then backward and left then right
    cancel out, so the bot ends where it started. Returns the number of problems found."""
    m = hw.m
    front, back, _, _ = hw.wall_room_cm()
    need = m.base_step + CHECK_MARGIN_CM
    if front >= need:
        order = (0, 1, 2, 3)
    elif back >= need:
        order = (1, 0, 2, 3)
    else:
        log(f"Movement check: no room to drive (front {front:.0f} cm, back {back:.0f} cm). "
            "Put the bot in the middle of the desk.")
        return 1
    names = ("forward", "backward", "turn +", "turn -")
    problems = 0
    for a in order:
        if should_stop is not None and should_stop():
            break
        before = hw.pose()
        if before is None:
            log("Movement check: lost the bot marker.")
            return problems + 1
        hw.execute(a)
        time.sleep(pause)
        after = hw.pose()
        if after is None:
            log("Movement check: lost the bot marker.")
            return problems + 1
        (p0, t0), (p1, t1) = before, after
        along = float((p1 - p0) @ np.array([np.cos(t0), np.sin(t0)]))
        turned = float(np.rad2deg(wrap(t1 - t0)))
        flags = []
        if a in (0, 1):
            want = m.base_step if a == 0 else -m.base_step
            if along * want < 0:
                flags.append("moved the WRONG WAY: forward and backward are swapped")
            elif abs(along - want) > CHECK_STEP_TOL * m.base_step:
                flags.append(f"step length is off by more than {CHECK_STEP_TOL:.0%}")
            if abs(turned) > CHECK_TURN_TOL_DEG:
                flags.append(f"heading drifted {turned:+.0f} deg while driving")
            line = f"moved {along:+.1f} cm (expected {want:+.1f})"
        else:
            want = m.turn_deg if a == 2 else -m.turn_deg
            if turned * want < 0:
                flags.append("turned the WRONG WAY: the turn direction is wired backwards "
                             "(action 2 must make the camera heading increase)")
            elif abs(turned - want) > CHECK_TURN_TOL_DEG:
                flags.append(f"turn is off by more than {CHECK_TURN_TOL_DEG:.0f} deg")
            line = f"turned {turned:+.0f} deg (expected {want:+.0f})"
        problems += len(flags)
        log(f"Movement check, action {a} ({names[a]}): {line}"
            + ("  OK" if not flags else "  PROBLEM: " + "; ".join(flags)))
    log("Movement check passed." if problems == 0
        else f"Movement check found {problems} problem(s): fix them before using --drive.")
    return problems
