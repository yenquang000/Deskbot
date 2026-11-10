"""Run a policy on the robot (Raspberry Pi). Needs only NumPy.

Fill in `RobotHardware` for the real motors/sensors (Ben); everything else -- building
the 26-number observation, keeping the cleaned-area map, the edge guard, the stop
conditions -- is already the same code the model trained against.

  python pi_runner.py --policy deskbot_policy.npz          # on the robot
  python pi_runner.py --sim --policy deskbot_policy.npz    # check the loop against the sim
"""
import argparse
import time

import numpy as np

from baseline import EdgeShield, LawnmowerPolicy
from env import Measurements
from obs import build_obs, make_grid, mark_clean, wrap


class RobotHardware:
    """Implement these four methods for the real robot."""

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
    """Dead-reckons the pose from the actions it commanded and keeps the cleaned map."""

    def __init__(self, W, H, m=None, start_heading=0.0):
        self.m = m or Measurements()
        self.W, self.H = W, H
        self.pen_corner = np.array([W - self.m.pen_corner_inset, H - self.m.pen_corner_inset])
        self.CX, self.CY, self.coverable = make_grid(W, H, self.m.robot_radius, self.m.clean_radius)
        self.cleaned = np.zeros_like(self.coverable)
        self.pos = np.array([W / 2, H / 2])
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

    def coverage(self):
        return float((self.cleaned & self.coverable).sum()) / max(int(self.coverable.sum()), 1)

    def build(self, walls_cm, debris, pen):
        return build_obs(self.theta, self.pos, walls_cm, debris, pen, self.pen_corner,
                         self.cleaned, self.CX, self.CY, self.coverable, self.m.robot_radius)


def run(hw, policy, W, H, max_steps=600, coverage_goal=0.80, guard=True, start_heading=0.0,
        step_delay=0.0):
    ob = ObsBuilder(W, H, start_heading=start_heading)
    shield = EdgeShield() if guard else None
    if hasattr(policy, "reset"):
        policy.reset()
    steps = 0
    try:
        for steps in range(1, max_steps + 1):
            debris, pen = hw.detections()
            obs = ob.build(hw.wall_room_cm(), debris, pen)
            a = policy.act(obs) if hasattr(policy, "act") else policy.predict(obs)
            if shield:
                a = shield(obs, a)
            hw.execute(a)
            ob.moved(a)
            if ob.coverage() >= coverage_goal and debris is None and pen is None:
                break
            time.sleep(step_delay)
    finally:
        hw.stop()
    return dict(steps=steps, coverage=ob.coverage(),
                guard_interventions=shield.interventions if shield else 0)


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
        print(run(SimHardware(e), pol, e.W, e.H, guard=not a.no_guard, start_heading=e.theta),
              "| sim edge hits:", e.edge_hits)
    else:
        raise SystemExit("Implement RobotHardware for the real robot, then call run(hw, pol, W, H).")
