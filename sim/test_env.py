"""Sanity tests for the simulation. Run: python test_env.py"""
import numpy as np

from baseline import EdgeShield, LawnmowerPolicy
from env import IDX, OBS_DIM, DeskbotEnv


def test_obs_and_determinism():
    for stage in (1, 2, 3):
        a, b = DeskbotEnv(stage=stage), DeskbotEnv(stage=stage)
        oa, _ = a.reset(seed=3)
        ob, _ = b.reset(seed=3)
        assert oa.shape == (OBS_DIM,) and np.isfinite(oa).all()
        assert np.array_equal(oa, ob)
        rng = np.random.default_rng(0)
        for _ in range(50):
            act = int(rng.integers(4))
            oa, ra, *_ = a.step(act)
            ob, rb, *_ = b.step(act)
            assert np.array_equal(oa, ob) and ra == rb
            assert oa.min() >= -3 and oa.max() <= 3


def test_randomisation_changes_each_reset():
    e = DeskbotEnv()
    sizes = {round(e.reset(seed=i) and e.W, 3) for i in range(20)}
    assert len(sizes) > 10


def test_edge_ends_episode():
    e = DeskbotEnv(randomize=False)
    e.reset(seed=0)
    for _ in range(100):
        _, r, term, _, info = e.step(0)
        if term:
            break
    assert term and info["edge_hit"] and r == -30.0


def test_coverage_monotone_and_bounded():
    e = DeskbotEnv(stage=1)
    e.reset(seed=1)
    last = e.coverage()
    for _ in range(60):
        _, _, term, _, info = e.step(int(e.np_random.integers(4)))
        assert info["coverage"] >= last and info["coverage"] <= 1.0
        last = info["coverage"]
        if term:
            break


def test_debris_pickup_and_pen_delivery():
    e = DeskbotEnv(stage=3, randomize=False)
    e.reset(seed=0)
    e.theta = 0.0
    e.debris = [e.pos + np.array([4.0, 0.0])]
    _, r, *_ = e.step(0)
    assert e.debris == [] and r > 4.0
    # put a pen right in front, with the corner just beyond it, and push it in
    e.pos = np.array([e.W - 40.0, e.pen_corner[1]])
    e.theta = 0.0
    e.pens = [e.pos + np.array([12.0, 0.0])]
    delivered = False
    for _ in range(10):
        e.step(0)
        if not e.pens:
            delivered = True
            break
    assert delivered


def test_shaping_cannot_be_farmed():
    """Going toward then away from debris must net ~0 shaping."""
    e = DeskbotEnv(stage=2, randomize=False)
    e.reset(seed=0)
    e.debris = [np.array([100.0, e.pos[1]])]
    e.theta = 0.0
    shaped = 0.0
    for a in (0, 1, 0, 1, 0, 1):
        e.cleaned[:] = True  # isolate shaping from coverage reward
        _, r, *_ = e.step(a)
        shaped += r + 0.02
    assert abs(shaped) < 0.05


def test_backup_pattern_is_safe():
    e = DeskbotEnv(stage=3)
    for ep in range(30):
        obs, _ = e.reset(seed=500 + ep)
        pol = LawnmowerPolicy()
        done = False
        while not done:
            obs, _, term, trunc, info = e.step(pol.predict(obs))
            done = term or trunc
        assert not info["edge_hit"], f"episode {ep} fell off"


def test_shield_blocks_driving_off():
    e = DeskbotEnv(stage=1)
    sh = EdgeShield()
    obs, _ = e.reset(seed=2)
    for _ in range(300):
        obs, _, term, trunc, info = e.step(sh(obs, 0))  # always "drive forward"
        assert not info["edge_hit"]
        if term or trunc:
            break
    assert sh.interventions > 0


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
