"""Run a policy on the robot (Raspberry Pi). Needs only NumPy.

`RobotHardware` is the robot side. The real one is `VisionHardware` in
vision_hardware.py (camera pose + Pico commands), started by the vision program:
`python main.py --drive ../sim/deskbot_policy.npz` in vision/. Everything else --
building the 26-number observation, keeping the cleaned-area map, the edge guard, the
stop conditions -- is already the same code the model trained against.

  python pi_runner.py --sim --policy deskbot_policy.npz    # check the loop against the sim
"""
import argparse
import time

import numpy as np

from baseline import EdgeShield, LawnmowerPolicy
from env import Measurements
from obs import build_obs, make_grid, mark_clean, wrap


class RobotHardware:
    """Implement these four methods for the real robot.

    Optional extras that run() uses when they exist: pose() -> ((x, y) cm, heading rad)
    or None, measured from outside (the camera), which replaces dead reckoning;
    wait_ready(should_stop, timeout) to wait for the pose to come back; clean(ms)."""

    def wall_room_cm(self):
        """[front, back, left, right]: how far the robot can still travel in each direction
        (relative to its heading) before its edge reaches the tape. Convert from the
        range / tape sensors using the mounting geometry."""
        raise NotImplementedError

    def detections(self):
        """(debris_xy, pen_xy) in desk coordinates (cm): nearest debris and nearest pen
        seen by the camera, either may be None."""
        raise NotImplementedError

    def execute(self, action):
        """0 forward one step, 1 backward one step, 2 turn left, 3 turn right.
        Block until the move finishes."""
        raise NotImplementedError

    def stop(self):
        """Stop all motors (also called on exit/exception)."""


class ObsBuilder:
    """Dead-reckons the pose from the actions it commanded (or takes a measured one
    through set_pose) and keeps the cleaned map."""

    def __init__(self, W, H, m=None, start_heading=0.0, start_pos=None):
        self.m = m or Measurements()
        self.W, self.H = W, H
        self.pen_corner = np.array([W - self.m.pen_corner_inset, H - self.m.pen_corner_inset])
        self.CX, self.CY, self.coverable = make_grid(W, H, self.m.robot_radius, self.m.clean_radius)
        self.cleaned = np.zeros_like(self.coverable)
        self.pos = np.array([W / 2, H / 2]) if start_pos is None else np.asarray(start_pos, float)
        self.theta = start_heading
        mark_clean(self.cleaned, self.CX, self.CY, self.coverable, self.pos, self.m.clean_radius)

    def moved(self, action):
        fwd = np.array([np.cos(self.theta), np.sin(self.theta)])
        if action == 0:
            self.pos = self.pos + self.m.base_step * fwd
        elif action == 1:
            self.pos = self.pos - self.m.base_step * fwd
        elif action == 2:
            self.theta = wrap(self.theta + np.deg2rad(self.m.turn_deg))
        elif action == 3:
            self.theta = wrap(self.theta - np.deg2rad(self.m.turn_deg))
        mark_clean(self.cleaned, self.CX, self.CY, self.coverable, self.pos, self.m.clean_radius)

    def set_pose(self, pos, theta):
        """Use a measured pose instead of dead reckoning, and mark the spot as cleaned."""
        self.pos, self.theta = np.asarray(pos, float), float(theta)
        mark_clean(self.cleaned, self.CX, self.CY, self.coverable, self.pos, self.m.clean_radius)

    def coverage(self):
        return float((self.cleaned & self.coverable).sum()) / max(int(self.coverable.sum()), 1)

    def build(self, walls_cm, debris, pen):
        return build_obs(self.theta, self.pos, walls_cm, debris, pen, self.pen_corner,
                         self.cleaned, self.CX, self.CY, self.coverable, self.m.robot_radius)


def _measured_pose(hw, should_stop, lost_timeout):
    """The hardware's pose. If it is missing: stop, wait for it, None after the timeout."""
    pose = hw.pose()
    if pose is None:
        hw.stop()
        if hw.wait_ready(should_stop, lost_timeout):
            pose = hw.pose()
    return pose


def run(hw, policy, W, H, max_steps=600, coverage_goal=0.80, guard=True, start_heading=0.0,
        step_delay=0.0, should_stop=None, clean_on_arrival=False, clean_ms=1500,
        lost_timeout=10.0):
    """The control loop. Returns steps, coverage, guard_interventions and `reason`:
    "goal reached", "max steps", "bot lost" or "stopped".

    If `hw` has pose(), the position and heading come from it before every step and the
    cleaned map is marked at the measured pose after every move (no dead reckoning).
    should_stop: called every step; the run ends when it returns True.
    clean_on_arrival: after a move that ends within robot_radius of the debris seen
    before the move, call hw.clean(clean_ms). The pre-move position is used because
    debris disappears from the camera once the bot covers it."""
    measured = hasattr(hw, "pose")
    stopping = should_stop or (lambda: False)
    shield = EdgeShield() if guard else None
    if hasattr(policy, "reset"):
        policy.reset()
    steps, reason, ob = 0, "max steps", None
    try:
        pose = _measured_pose(hw, should_stop, lost_timeout) if measured else None
        if measured and pose is None:
            reason = "stopped" if stopping() else "bot lost"
        else:
            ob = ObsBuilder(W, H, start_heading=start_heading,
                            start_pos=pose[0] if measured else None)
        while ob is not None and steps < max_steps:
            if stopping():
                reason = "stopped"
                break
            if measured:
                pose = _measured_pose(hw, should_stop, lost_timeout)
                if pose is None:
                    reason = "stopped" if stopping() else "bot lost"
                    break
                ob.set_pose(*pose)
            debris, pen = hw.detections()
            obs = ob.build(hw.wall_room_cm(), debris, pen)
            a = policy.act(obs) if hasattr(policy, "act") else policy.predict(obs)
            if shield:
                a = shield(obs, a)
            hw.execute(a)
            steps += 1
            if measured:
                pose = hw.pose()  # if it is None, the next step waits for it
                if pose is not None:
                    ob.set_pose(*pose)
                    if (clean_on_arrival and debris is not None
                            and np.linalg.norm(pose[0] - debris) < ob.m.robot_radius):
                        hw.clean(clean_ms)
            else:
                ob.moved(a)
            if ob.coverage() >= coverage_goal and debris is None and pen is None:
                reason = "goal reached"
                break
            time.sleep(step_delay)
    finally:
        hw.stop()
    return dict(steps=steps, coverage=ob.coverage() if ob else 0.0,
                guard_interventions=shield.interventions if shield else 0, reason=reason)


class SimHardware(RobotHardware):
    """Lets the runner drive the simulator through the same interface as the robot."""

    def __init__(self, env):
        self.env = env

    def wall_room_cm(self):
        e = self.env
        return [e._clear(e.theta + o) + e.np_random.normal(0, e.sensor_noise)
                for o in (0.0, np.pi, np.pi / 2, -np.pi / 2)]

    def detections(self):
        e = self.env
        deb = min(e.debris, key=lambda d: np.linalg.norm(d - e.pos), default=None)
        pen = min(e.pens, key=lambda p: np.linalg.norm(p - e.pos), default=None)
        return deb, pen

    def execute(self, action):
        self.env.step(action)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default=None, help=".npz from export_to_pi.py (default: backup pattern)")
    ap.add_argument("--sim", action="store_true")
    ap.add_argument("--no-guard", action="store_true")
    a = ap.parse_args()
    if a.policy:
        from pi_inference import DeskbotPolicy
        pol = DeskbotPolicy(a.policy)
    else:
        pol = LawnmowerPolicy()
    if a.sim:
        from env import DeskbotEnv
        e = DeskbotEnv(stage=3)
        e.reset(seed=1)
        result = run(SimHardware(e), pol, e.W, e.H, guard=not a.no_guard, start_heading=e.theta)
        result.pop("reason")  # keeps this line the same as before `reason` was added
        print(result, "| sim edge hits:", e.edge_hits)
    else:
        raise SystemExit("On the robot the loop runs inside the vision program:\n"
                         "  cd vision && python main.py --drive ../sim/deskbot_policy.npz")
