"""Deskbot desk-cleaning simulation (Gymnasium environment).

Stages (curriculum, matches the plan):
  1  coverage only                                  (Days 1-2)
  2  + debris with imperfect camera detections      (Days 3-4)
  3  + pens pushed into the pen corner              (Day 5 onward)

Units are centimetres. Domain randomisation (desk size, grip, turn error, sensor
noise, camera quality) is re-drawn on EVERY reset, not once at construction.

Nico's real measurements go in `Measurements` (Day 6). The defaults are the numbers
that were in the first version of env.py -- replace them.
"""
from dataclasses import dataclass

import numpy as np

from obs import (ACTION_NAMES, IDX, OBS_DIM, OBS_NAMES, build_obs, make_grid,  # noqa: F401
                 mark_clean, push_position, wrap)

try:  # real Gymnasium when installed (always the case for training)
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # tiny fallback so the sim can be tested without Gymnasium
    class _Discrete:
        def __init__(self, n):
            self.n = n

    class _Box:
        def __init__(self, low, high, dtype=np.float32):
            self.low, self.high, self.dtype, self.shape = low, high, dtype, np.shape(low)

    class spaces:  # noqa: N801
        Discrete = _Discrete
        Box = _Box

    class gym:  # noqa: N801
        class Env:
            def reset(self, seed=None, options=None):
                if seed is not None or not hasattr(self, "np_random"):
                    self.np_random = np.random.default_rng(seed)


@dataclass
class Measurements:
    """Replace with Nico's real numbers (Day 6)."""
    desk_width: float = 120.65      # cm
    desk_height: float = 59.69      # cm
    base_step: float = 5.0          # cm per forward/backward action
    robot_radius: float = 10.0      # cm
    turn_deg: float = 45.0          # degrees per turn action
    clean_radius: float = 10.0      # cm swept around the robot centre
    pen_corner_inset: float = 15.0  # pen target point is this far from both walls


@dataclass
class Randomization:
    desk_scale: tuple = (0.95, 1.05)
    grip: tuple = (0.85, 1.0)          # per-episode step-length factor
    step_jitter: float = 0.03          # per-step multiplicative noise
    turn_noise: float = 0.05           # relative std of turn angle error
    sensor_noise: tuple = (0.3, 1.0)   # cm, wall-distance sensor std
    heading_noise: float = 0.02        # rad, observed heading std
    pos_noise: float = 1.0             # cm, odometry error on observed position
    detect_prob: tuple = (0.80, 0.98)  # camera sees a debris piece in range
    detect_noise: float = 2.0          # cm, detection position std
    false_pos: float = 0.02            # chance per step of a phantom detection
    detect_range: float = 90.0


class DeskbotEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, stage=3, measurements=None, rand=None, randomize=True,
                 max_steps=600, coverage_goal=0.80, n_debris=10, n_pens=2, cell=5.0):
        super().__init__()
        self.stage = stage
        self.m = measurements or Measurements()
        self.rand = rand or Randomization()
        self.randomize = randomize
        self.max_steps = max_steps
        self.coverage_goal = coverage_goal
        self.n_debris = n_debris if stage >= 2 else 0
        self.n_pens = n_pens if stage >= 3 else 0
        self.cell = cell

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=np.full(OBS_DIM, -3.0, np.float32),
            high=np.full(OBS_DIM, 3.0, np.float32), dtype=np.float32)

    # ------------------------------------------------------------------ reset
    def _draw_episode_params(self):
        r, m, g = self.rand, self.m, self.np_random
        if self.randomize:
            s = g.uniform(*r.desk_scale)
            self.grip = g.uniform(*r.grip)
            self.sensor_noise = g.uniform(*r.sensor_noise)
            self.detect_prob = g.uniform(*r.detect_prob)
        else:
            s, self.grip = 1.0, 1.0
            self.sensor_noise, self.detect_prob = 0.5, 0.95
        self.W, self.H = m.desk_width * s, m.desk_height * s
        self.pen_corner = np.array([self.W - m.pen_corner_inset, self.H - m.pen_corner_inset])

    def _rand_pos(self, margin):
        g = self.np_random
        return np.array([g.uniform(margin, self.W - margin), g.uniform(margin, self.H - margin)])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._draw_episode_params()
        self.CX, self.CY, self.coverable = make_grid(
            self.W, self.H, self.m.robot_radius, self.m.clean_radius, self.cell)
        self.cleaned = np.zeros_like(self.coverable)
        self.n_coverable = int(self.coverable.sum())
        self.t = 0
        self.edge_hits = 0
        g = self.np_random
        self.pos = np.array([self.W / 2, self.H / 2])
        # placed by hand: heading is a multiple of the turn increment
        self.theta = float(wrap(g.integers(8) * np.deg2rad(self.m.turn_deg))) if self.randomize else 0.0
        mg = self.m.robot_radius + 2.0
        self.debris = [self._rand_pos(mg) for _ in range(self.n_debris)]
        self.pens = []
        while len(self.pens) < self.n_pens:
            p = self._rand_pos(20.0)
            if np.linalg.norm(p - self.pen_corner) > 35.0:
                self.pens.append(p)
        mark_clean(self.cleaned, self.CX, self.CY, self.coverable, self.pos, self.m.clean_radius)
        return self._get_obs(), {}

    # -------------------------------------------------------------- geometry
    def _clear(self, ang):
        """How far the robot can travel along `ang` before its footprint crosses an
        edge (exact travel distance, so it is also correct for oblique headings)."""
        r = self.m.robot_radius
        c, s = np.cos(ang), np.sin(ang)
        tx = (self.W - r - self.pos[0]) / c if c > 1e-9 else ((r - self.pos[0]) / c if c < -1e-9 else np.inf)
        ty = (self.H - r - self.pos[1]) / s if s > 1e-9 else ((r - self.pos[1]) / s if s < -1e-9 else np.inf)
        return max(min(tx, ty), 0.0)

    def _off_desk(self):
        r = self.m.robot_radius
        return (self.pos[0] - r < 0 or self.pos[0] + r > self.W or
                self.pos[1] - r < 0 or self.pos[1] + r > self.H)

    def coverage(self):
        return float((self.cleaned & self.coverable).sum()) / max(self.n_coverable, 1)

    # ---------------------------------------------------------- observations
    def _get_obs(self):
        """Simulate the sensors (noise, misses, phantoms), then use the shared builder."""
        g, r = self.np_random, self.rand
        th = self.theta + (g.normal(0, r.heading_noise) if self.randomize else 0.0)
        est = self.pos + (g.normal(0, r.pos_noise, 2) if self.randomize else 0.0)
        walls = [self._clear(self.theta + off) + g.normal(0, self.sensor_noise)
                 for off in (0.0, np.pi, np.pi / 2, -np.pi / 2)]

        best = None  # nearest *detected* debris
        if self.stage >= 2:
            for d in self.debris:
                if np.linalg.norm(d - self.pos) < r.detect_range and g.random() < self.detect_prob:
                    cand = d + g.normal(0, r.detect_noise, 2)
                    if best is None or np.linalg.norm(cand - est) < np.linalg.norm(best - est):
                        best = cand
            if g.random() < r.false_pos:
                ph = self._rand_pos(10.0)
                if best is None or np.linalg.norm(ph - est) < np.linalg.norm(best - est):
                    best = ph
        pen = None
        if self.pens:
            pen = min(self.pens, key=lambda q: np.linalg.norm(q - self.pos))
            pen = pen + (g.normal(0, r.detect_noise, 2) if self.randomize else 0.0)
        return build_obs(th, est, walls, best, pen, self.pen_corner, self.cleaned,
                         self.CX, self.CY, self.coverable, self.m.robot_radius)

    # ------------------------------------------------------------------ step
    def _objective(self):
        """Target for reward shaping (ground truth, not what the policy sees). Returns
        (kind, array) -- the array object itself, to tell later if it still exists."""
        if self.debris:
            return ("debris", min(self.debris, key=lambda d: np.linalg.norm(d - self.pos)))
        if self.pens:
            return ("pen", min(self.pens, key=lambda p: np.linalg.norm(p - self.pos)))
        return None

    def _potential(self, kind, tgt):
        if kind == "debris":
            return np.linalg.norm(tgt - self.pos), 0.0
        return (np.linalg.norm(push_position(tgt, self.pen_corner, self.m.robot_radius) - self.pos),
                np.linalg.norm(self.pen_corner - tgt))

    def step(self, action):
        g, m, r = self.np_random, self.m, self.rand
        self.t += 1
        obj = self._objective()
        before = self._potential(*obj) if obj else None

        step_len = m.base_step * self.grip
        if self.randomize:
            step_len *= 1 + g.normal(0, r.step_jitter)
        turn = np.deg2rad(m.turn_deg) * (1 + (g.normal(0, r.turn_noise) if self.randomize else 0.0))
        fwd = np.array([np.cos(self.theta), np.sin(self.theta)])
        if action == 0:
            self.pos = self.pos + step_len * fwd
        elif action == 1:
            self.pos = self.pos - step_len * fwd
        elif action == 2:
            self.theta = float(wrap(self.theta + turn))
        elif action == 3:
            self.theta = float(wrap(self.theta - turn))

        reward, terminated, success = -0.02, False, False

        if self._off_desk():
            self.edge_hits += 1
            return self._get_obs(), -30.0, True, False, self._info(edge_hit=True, success=False)

        reward += 0.2 * mark_clean(self.cleaned, self.CX, self.CY, self.coverable, self.pos, m.clean_radius)

        for i, d in enumerate(self.debris):
            if np.linalg.norm(d - self.pos) < m.robot_radius:
                self.debris.pop(i)
                reward += 5.0
                break

        if action == 0:  # driving forward pushes a pen that is in front of the robot
            for i, p in enumerate(self.pens):
                v = p - self.pos
                dist = np.linalg.norm(v)
                if 1e-6 < dist < m.robot_radius + 3.0 and (v @ fwd) / dist > 0.5:
                    p[:] = np.clip(self.pos + (m.robot_radius + 1.5) * fwd,
                                   [1.0, 1.0], [self.W - 1.0, self.H - 1.0])
                    if np.linalg.norm(p - self.pen_corner) < 15.0:
                        self.pens.pop(i)
                        reward += 15.0
                    break

        # Potential-based shaping toward the current objective: it differences the same
        # target before/after the move, so oscillating cannot farm reward.
        if obj:
            kind, tgt = obj
            if any(x is tgt for x in (self.debris if kind == "debris" else self.pens)):
                after = self._potential(kind, tgt)
                reward += 0.05 * (before[0] - after[0]) + 0.1 * (before[1] - after[1])

        if self.coverage() >= self.coverage_goal and not self.debris and not self.pens:
            reward += 30.0
            terminated = success = True
        truncated = (not terminated) and self.t >= self.max_steps
        return self._get_obs(), float(reward), terminated, truncated, self._info(
            edge_hit=False, success=success)

    def _info(self, **extra):
        extra.update(coverage=self.coverage(), debris_left=len(self.debris),
                     pens_left=len(self.pens), edge_hits=self.edge_hits)
        return extra
