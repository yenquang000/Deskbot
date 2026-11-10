"""Observation layout + builder shared by the simulation and the Pi runner.

The simulation feeds this *noisy measurements*; the Pi feeds it real sensor
readings. Using one function for both means the model sees identically
constructed inputs in training and on the robot.
"""
import numpy as np

OBS_DIM = 26
OBS_NAMES = [
    "sin_heading", "cos_heading",
    "wall_front", "wall_back", "wall_left", "wall_right",          # travel room cm / 100
    "debris_seen", "debris_dist", "debris_sin", "debris_cos",
    "pen_seen", "pen_dist", "pen_sin", "pen_cos",
    "pushpos_dist", "pushpos_sin", "pushpos_cos", "pen_to_corner_dist",
    "coverage",
    "unclean_ahead", "unclean_behind", "unclean_left", "unclean_right",
    "nearest_unclean_dist", "nearest_unclean_sin", "nearest_unclean_cos",
]
assert len(OBS_NAMES) == OBS_DIM
IDX = {n: i for i, n in enumerate(OBS_NAMES)}
ACTION_NAMES = ["forward", "backward", "turn_left", "turn_right"]


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def make_grid(W, H, robot_radius, clean_radius, cell=5.0):
    """Cleaned-area grid. Returns (CX, CY, coverable) -- cell centres and the mask of
    cells the robot can actually reach (corners of the desk are not coverable)."""
    nx, ny = int(np.ceil(W / cell)), int(np.ceil(H / cell))
    CX, CY = np.meshgrid((np.arange(nx) + 0.5) * cell, (np.arange(ny) + 0.5) * cell, indexing="ij")
    dx = np.maximum.reduce([robot_radius - CX, np.zeros_like(CX), CX - (W - robot_radius)])
    dy = np.maximum.reduce([robot_radius - CY, np.zeros_like(CY), CY - (H - robot_radius)])
    coverable = (np.hypot(dx, dy) <= clean_radius) & (CX <= W) & (CY <= H)
    return CX, CY, coverable


def mark_clean(cleaned, CX, CY, coverable, pos, clean_radius):
    """Mark cells swept at `pos` as cleaned (in place). Returns number of new cells."""
    d2 = (CX - pos[0]) ** 2 + (CY - pos[1]) ** 2
    new = (d2 <= clean_radius ** 2) & coverable & ~cleaned
    cleaned |= new
    return int(new.sum())


def push_position(pen, pen_corner, robot_radius):
    """Where the robot should stand to push `pen` straight toward the corner."""
    u = pen_corner - pen
    n = np.linalg.norm(u)
    u = u / n if n > 1e-6 else np.array([1.0, 0.0])
    return pen - u * (robot_radius + 1.5)


def _rel(est, target, theta):
    v = target - est
    return np.linalg.norm(v), wrap(np.arctan2(v[1], v[0]) - theta)


def build_obs(theta, est, walls_cm, debris, pen, pen_corner, cleaned, CX, CY, coverable,
              robot_radius):
    """theta: heading (rad). est: believed (x, y) cm. walls_cm: travel room [front, back,
    left, right] in cm. debris / pen: believed (x, y) of the nearest piece or None."""
    o = np.zeros(OBS_DIM, np.float32)
    o[0], o[1] = np.sin(theta), np.cos(theta)
    o[2:6] = np.clip(np.asarray(walls_cm, float), 0, 150) / 100
    if debris is not None:
        dist, ang = _rel(est, debris, theta)
        o[6], o[7], o[8], o[9] = 1, min(dist / 100, 2), np.sin(ang), np.cos(ang)
    if pen is not None:
        dist, ang = _rel(est, pen, theta)
        o[10], o[11], o[12], o[13] = 1, min(dist / 100, 2), np.sin(ang), np.cos(ang)
        dist, ang = _rel(est, push_position(pen, pen_corner, robot_radius), theta)
        o[14], o[15], o[16] = min(dist / 100, 2), np.sin(ang), np.cos(ang)
        o[17] = min(np.linalg.norm(pen_corner - pen) / 100, 2)
    unclean = coverable & ~cleaned
    o[18] = (cleaned & coverable).sum() / max(coverable.sum(), 1)
    for k, off in enumerate((0.0, np.pi, np.pi / 2, -np.pi / 2)):
        pp = est + 20.0 * np.array([np.cos(theta + off), np.sin(theta + off)])
        near = ((CX - pp[0]) ** 2 + (CY - pp[1]) ** 2 < 100.0) & coverable
        o[19 + k] = (near & unclean).sum() / max(near.sum(), 1)
    if unclean.any():
        d2 = np.where(unclean, (CX - est[0]) ** 2 + (CY - est[1]) ** 2, np.inf)
        i = np.unravel_index(np.argmin(d2), d2.shape)
        dist, ang = _rel(est, np.array([CX[i], CY[i]]), theta)
        o[23], o[24], o[25] = min(dist / 100, 2), np.sin(ang), np.cos(ang)
    return o
