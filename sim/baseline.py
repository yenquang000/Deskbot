"""Backup pattern (lawnmower sweep) and the edge-guard safety shield.

Both work purely from the observation vector, so they run unchanged on the Pi.
"""
import numpy as np

from env import IDX

FWD, BACK, LEFT, RIGHT = 0, 1, 2, 3


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def _heading(obs):
    return float(np.arctan2(obs[IDX["sin_heading"]], obs[IDX["cos_heading"]]))


def _turn_toward(theta, target, tol=0.4):  # turns come in 45 deg (0.79 rad) steps
    d = _wrap(target - theta)
    if abs(d) < tol:
        return None
    return LEFT if d > 0 else RIGHT


class LawnmowerPolicy:
    """Drive to the bottom-left, then sweep lanes along x, shifting up by `lane` cm.
    Stops `margin` cm (clear distance) before an edge so sensor noise can't push it over."""

    def __init__(self, margin=8.0, lane_steps=3):
        self.margin = margin / 100.0
        self.lane_steps = lane_steps
        self.reset()

    def reset(self):
        self.phase, self.xdir, self.ydir, self.shift_count = "bottom", 0.0, 1, 0

    def predict(self, obs):
        th, front = _heading(obs), obs[IDX["wall_front"]]
        if self.phase == "bottom":          # face -y and drive to the bottom wall
            a = _turn_toward(th, -np.pi / 2)
            if a is not None:
                return a
            if front > self.margin:
                return FWD
            self.phase = "left"
        if self.phase == "left":            # face -x and drive to the left wall
            a = _turn_toward(th, np.pi)
            if a is not None:
                return a
            if front > self.margin:
                return FWD
            self.phase, self.xdir = "lane", 0.0
        if self.phase == "lane":
            a = _turn_toward(th, self.xdir)
            if a is not None:
                return a
            if front > self.margin:
                return FWD
            self.phase, self.shift_count = "shift", 0
        if self.phase == "shift":
            a = _turn_toward(th, self.ydir * np.pi / 2)
            if a is not None:
                return a
            if front > self.margin and self.shift_count < self.lane_steps:
                self.shift_count += 1
                return FWD
            if front <= self.margin:        # hit the top/bottom: sweep back the other way
                self.ydir *= -1
            self.xdir = np.pi if self.xdir == 0.0 else 0.0
            self.phase = "lane"
            return self.predict(obs)
        raise RuntimeError(self.phase)


class EdgeShield:
    """Last line of defence: overrides any policy action that would drive the robot
    off the desk given the wall sensors. Costs a little coverage near the edges."""

    def __init__(self, margin=8.0):
        self.margin = margin / 100.0
        self.interventions = 0

    def __call__(self, obs, action):
        if action == FWD and obs[IDX["wall_front"]] <= self.margin:
            self.interventions += 1
            return LEFT if obs[IDX["wall_left"]] >= obs[IDX["wall_right"]] else RIGHT
        if action == BACK and obs[IDX["wall_back"]] <= self.margin:
            self.interventions += 1
            return LEFT if obs[IDX["wall_left"]] >= obs[IDX["wall_right"]] else RIGHT
        return action
