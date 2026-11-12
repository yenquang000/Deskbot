"""End-to-end check of the vision -> policy bridge without a camera or a Pico.

A fake link gives VisionHardware the simulator's state in vision's format (mm, degrees
0-359, object boxes) and turns the Pico commands into env.step() calls.

Run: python test_vision_bridge.py
"""
import numpy as np

from env import DeskbotEnv
from goto_marker import ARRIVE_CM, goto_marker, next_action
from obs import travel_room
from pi_inference import DeskbotPolicy
from pi_runner import ObsBuilder, run
from vision_hardware import VisionHardware, is_pen, movement_check


class FakeLink:
    """Same latest / send_command / wait_done interface as PicoLink, backed by a DeskbotEnv."""

    def __init__(self, env):
        self.env = env
        self.commands = []
        self.visible = True     # False = the bot marker is hidden
        self.flip_turns = False  # True = turns wired backwards
        self.answer = True       # False = the Pico never says DONE
        self.target_cm = None    # (x, y) of the target marker, or None = not seen

    def latest(self):
        e = self.env
        bot = {"x": e.pos[0] * 10, "y": e.pos[1] * 10, "hd": np.rad2deg(e.theta) % 360}
        objs = []
        for d in e.debris:   # small squares
            objs.append({"id": len(objs), "x": d[0] * 10, "y": d[1] * 10, "w": 15, "h": 15, "a": 225})
        for p in e.pens:     # about 140 x 10 mm
            objs.append({"id": len(objs), "x": p[0] * 10, "y": p[1] * 10, "w": 140, "h": 10, "a": 1400})
        return ((e.W * 10, e.H * 10), True, bot if self.visible else None, objs), 0.0

    def latest_target(self):
        t = self.target_cm
        return (None if t is None else {"x": t[0] * 10, "y": t[1] * 10}), 0.0

    def send_command(self, do, a=None):
        self.commands.append((do, a))
        if do == "move" and self.answer:
            self.env.step({2: 3, 3: 2}.get(a, a) if self.flip_turns else a)
        return len(self.commands)

    def wait_done(self, cmd_id, timeout=5.0):
        return self.answer


def make(seed, **env_kwargs):
    env = DeskbotEnv(stage=3, **env_kwargs)
    env.reset(seed=seed)
    link = FakeLink(env)
    return env, link, VisionHardware(link, log=lambda text: None, settle=0.0)


def test_travel_room_matches_env():
    e = DeskbotEnv()
    rng = np.random.default_rng(0)
    for i in range(2000):
        if i % 100 == 0:
            e.reset(seed=i)
        e.pos = np.array([rng.uniform(-5, e.W + 5), rng.uniform(-5, e.H + 5)])
        ang = rng.uniform(-7, 7) if i % 4 else rng.integers(-8, 9) * np.pi / 4
        assert travel_room(e.pos, ang, e.W, e.H, e.m.robot_radius) == e._clear(ang)


def test_pen_classifier():
    assert is_pen({"w": 140, "h": 10, "a": 1400})        # along an axis
    assert is_pen({"w": 10, "h": 140, "a": 1400})
    assert is_pen({"w": 105, "h": 105, "a": 1500})       # diagonal: square box, mostly empty
    assert not is_pen({"w": 15, "h": 15, "a": 225})      # crumb
    assert not is_pen({"w": 60, "h": 50, "a": 2400})     # blob
    assert not is_pen({"w": 30, "h": 8, "a": 240})       # thin but too short


def test_conversions_match_env():
    for seed in range(5):
        e, link, hw = make(seed)
        rng = np.random.default_rng(seed)
        for _ in range(40):
            pos, theta = hw.pose()
            assert np.allclose(pos, e.pos) and abs(np.exp(1j * theta) - np.exp(1j * e.theta)) < 1e-9
            assert np.allclose(hw.desk_cm(), (e.W, e.H))
            walls = [e._clear(e.theta + o) for o in (0.0, np.pi, np.pi / 2, -np.pi / 2)]
            assert np.allclose(hw.wall_room_cm(), walls, atol=1e-6)
            debris, pen = hw.detections()
            near = lambda q: np.linalg.norm(q - e.pos)  # noqa: E731
            want_debris = min(e.debris, key=near, default=None)
            want_pen = min(e.pens, key=near, default=None)
            assert (debris is None) == (want_debris is None) and (pen is None) == (want_pen is None)
            assert debris is None or np.allclose(debris, want_debris)
            assert pen is None or np.allclose(pen, want_pen)
            a = int(rng.integers(4))
            if a < 2 and walls[a] < 8:
                a = 2
            hw.execute(a)


def test_pose_missing_and_stale():
    e, link, hw = make(0)
    link.visible = False
    assert hw.pose() is None and hw.detections() == (None, None)
    assert hw.wait_ready(timeout=0.2) is False
    assert hw.wait_ready(should_stop=lambda: True) is False
    link.visible = True
    state = link.latest()[0]
    link.latest = lambda: (state, 2.0)   # two seconds old
    assert hw.pose() is None


def test_desk_warning():
    e, link, hw = make(0, randomize=False)
    assert hw.desk_warning() is None
    e.W = 100.0   # the vision default, 1000 mm
    assert "retrain" in hw.desk_warning()


def test_execute_times_out_and_stop_never_fails():
    e, link, hw = make(0)
    link.answer = False
    try:
        hw.execute(0)
        raise AssertionError("expected a TimeoutError")
    except TimeoutError:
        pass
    link.send_command = None  # a link that is completely gone
    hw.stop()


def test_policy_drives_through_the_bridge():
    """The exported policy + edge guard, on the camera pose: never off the desk."""
    policy = DeskbotPolicy("deskbot_policy.npz")
    coverages = []
    for seed in range(1, 7):
        e, link, hw = make(seed)
        start, _ = hw.pose()
        result = run(hw, policy, *hw.desk_cm(), start_heading=hw.pose()[1])
        assert e.edge_hits == 0, f"seed {seed} left the desk"
        assert result["reason"] in ("goal reached", "max steps")
        assert link.commands[-1][0] == "stop"
        # the runner's cleaned map is the env's: both are marked at the same poses
        assert abs(result["coverage"] - e.coverage()) < 1e-9
        coverages.append(result["coverage"])
        print(f"   seed {seed}: {result}  debris left {len(e.debris)}, pens left {len(e.pens)}")
    assert np.mean(coverages) > 0.5, coverages


def test_bot_lost_and_should_stop():
    policy = DeskbotPolicy("deskbot_policy.npz")
    e, link, hw = make(1)
    moves = []
    real = hw.execute

    def execute(action):
        real(action)
        moves.append(action)
        if len(moves) == 5:
            link.visible = False

    hw.execute = execute
    result = run(hw, policy, *hw.desk_cm(), lost_timeout=0.3)
    assert result["reason"] == "bot lost" and result["steps"] == 5
    assert link.commands[-1][0] == "stop"

    e, link, hw = make(1)
    result = run(hw, policy, *hw.desk_cm(), should_stop=lambda: len(link.commands) >= 3)
    assert result["reason"] == "stopped" and result["steps"] == 3


def test_clean_on_arrival():
    class Forward:
        def act(self, obs):
            return 0

    for flag, expected in ((False, 0), (True, 1)):
        e, link, hw = make(0, randomize=False)
        e.theta = 0.0
        e.debris = [e.pos + np.array([12.0, 0.0])]
        e.pens = []
        run(hw, Forward(), *hw.desk_cm(), max_steps=3, clean_on_arrival=flag, clean_ms=700)
        cleans = [c for c in link.commands if c[0] == "clean"]
        assert len(cleans) == expected and (not cleans or cleans[0] == ("clean", 700))


def test_set_pose_marks_cleaned():
    ob = ObsBuilder(120.0, 60.0, start_pos=(30.0, 30.0))
    assert np.allclose(ob.pos, (30.0, 30.0))
    before = ob.coverage()
    ob.set_pose((80.0, 30.0), 1.0)
    assert ob.coverage() > before and ob.theta == 1.0


def test_movement_check():
    e, link, hw = make(0, randomize=False)
    start = e.pos.copy(), e.theta
    lines = []
    assert movement_check(hw, lines.append, pause=0.0) == 0
    assert [c[1] for c in link.commands] == [0, 1, 2, 3]
    assert np.allclose(e.pos, start[0]) and abs(e.theta - start[1]) < 1e-9

    e, link, hw = make(0, randomize=False)
    link.flip_turns = True
    lines = []
    assert movement_check(hw, lines.append, pause=0.0) == 2
    assert any("wired backwards" in line for line in lines)

    e, link, hw = make(0, randomize=False)
    e.grip = 0.7  # 3.5 cm steps: 30% short
    lines = []
    assert movement_check(hw, lines.append, pause=0.0) == 2
    assert any("step length" in line for line in lines)


def test_goto_next_action():
    east = 0.0
    assert next_action((50, 30), east, (50 + ARRIVE_CM - 1, 30), 50) == "arrived"
    assert next_action((50, 30), east, (90, 30), 50) == 0             # straight ahead
    assert next_action((50, 30), east, (90, 45), 50) == 0             # 20 deg off: close enough
    assert next_action((50, 30), east, (50, 60), 50) == 2             # +y is +90 deg
    assert next_action((50, 30), east, (50, 0), 50) == 3              # -y is -90 deg
    assert next_action((50, 30), np.deg2rad(350), (90, 40), 50) == 2  # wraps past 0
    assert next_action((50, 30), east, (90, 30), 3.0) == "blocked"    # no room for a step


def test_goto_reaches_target():
    # Targets all over the desk, including behind the bot and near the edges.
    for seed, target in enumerate([(100, 45), (20, 15), (60, 50), (110, 12), (25, 48)]):
        e, link, hw = make(seed, randomize=False)
        link.target_cm = target
        result = goto_marker(hw, log=lambda text: None, follow=False, max_moves=150, poll=0)
        distance = np.linalg.norm(e.pos - target)
        assert result["reason"] == "arrived", (seed, target, result, e.pos)
        assert distance <= ARRIVE_CM and e.edge_hits == 0, (seed, target, distance)
        assert link.commands[-1][0] == "stop"


def test_goto_follows_a_moved_target():
    e, link, hw = make(1, randomize=False)
    link.target_cm = (30, 30)
    calls = []

    def should_stop():
        calls.append(1)
        if len(calls) == 60:  # partway through: move the marker across the desk
            link.target_cm = (95, 30)
        return len(calls) > 200
    result = goto_marker(hw, log=lambda text: None, should_stop=should_stop, poll=0)
    assert result["reason"] == "stopped"
    assert np.linalg.norm(e.pos - (95, 30)) <= ARRIVE_CM and e.edge_hits == 0


def test_goto_waits_without_target_or_bot():
    for hide in ("target", "bot"):
        e, link, hw = make(0, randomize=False)
        link.target_cm = None if hide == "target" else (90, 30)
        link.visible = hide != "bot"
        lines, calls = [], []
        result = goto_marker(hw, log=lines.append, poll=0,
                             should_stop=lambda: calls.append(1) or len(calls) > 20)
        assert result == {"reason": "stopped", "moves": 0} and link.commands == []
        assert len(lines) == 1 and hide in lines[0], lines  # logged once, not every poll


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
