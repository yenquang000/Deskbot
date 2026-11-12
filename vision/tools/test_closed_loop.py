"""
Closed-loop check of the whole drive chain, with no camera, Pico or motors:

  simulated bot -> rendered camera image with the ArUco markers -> vision worker
  (markers, desk mm) -> VisionHardware -> --goto / --drive-check / --drive
  -> messaging.encode -> the real Pico program (parse, on_update, timed moves,
  ramps, TB6612 pins) -> fake machine.Pin / PWM -> wheel speeds -> simulated bot

The Pico program runs unchanged on a virtual clock. The wheel speeds come from
the TB6612 pin states it sets (IN1/IN2 direction, PWM duty), so this checks the
pin numbers, the wheel directions, LEFT/RIGHT_REVERSED and the heading
convention against what the camera sees.

Run (from the vision folder):  python tools/test_closed_loop.py
Exits with 0 when every check passes.
"""

import math
import sys
import time
import types
from pathlib import Path

import numpy as np

VISION_DIR = Path(__file__).resolve().parent.parent
ROOT = VISION_DIR.parent
sys.path[:0] = [str(ROOT), str(VISION_DIR), str(VISION_DIR / "tools")]
sys.path.append(str(ROOT / "sim"))

# ----- fake MicroPython `machine` module, recording the pin states --------------


class Pin:
    OUT = 1
    pins = {}

    def __init__(self, number, mode=None, value=None):
        self.number, self.level = number, value or 0
        Pin.pins[number] = self

    def value(self, level=None):
        if level is None:
            return self.level
        self.level = level


class PWM:
    pwms = {}

    def __init__(self, pin):
        self.duty = 0
        PWM.pwms[pin.number] = self

    def freq(self, hz):
        self.hz = hz

    def duty_u16(self, duty):
        assert 0 <= duty <= 65535, duty
        self.duty = duty


sys.modules["machine"] = types.SimpleNamespace(Pin=Pin, PWM=PWM)

import test_coordinates as tc  # noqa: E402  (scene drawing, learned worker)
from hardware import messaging, pico_controller as pc  # noqa: E402
from shared import config as shared  # noqa: E402
from goto_marker import ARRIVE_CM, goto_marker  # noqa: E402
from pi_inference import DeskbotPolicy  # noqa: E402
from pi_runner import run  # noqa: E402
from vision_hardware import VisionHardware, movement_check  # noqa: E402

# The real default desk, and the scene module drawing it.
tc.DESK_W, tc.DESK_H = shared.DESK_WIDTH_MM, shared.DESK_HEIGHT_MM
W_CM, H_CM = tc.DESK_W / 10, tc.DESK_H / 10
ROBOT_RADIUS_CM = 10.0
results = []


def check(name, passed, detail=""):
    results.append(passed)
    print(f"{'PASS' if passed else 'FAIL'}  {name}{f'  ({detail})' if detail else ''}")


# ----- the simulated bot ----------------------------------------------------------

# Wheel speed at full duty, chosen so the Pico's estimates (CM_PER_S at
# DRIVE_SPEED, DEG_PER_S at TURN_SPEED) are exactly right when the errors are 1.
FULL_CM_S = pc.CM_PER_S / pc.DRIVE_SPEED
TRACK_CM = 2 * FULL_CM_S * pc.TURN_SPEED / math.radians(pc.DEG_PER_S)


class Bot:
    """Differential drive driven by the TB6612 pins the Pico program sets."""

    def __init__(self, x, y, heading_deg, speed_error=1.0, turn_error=1.0, wired=(1, 1)):
        self.x, self.y, self.th = x, y, math.radians(heading_deg)
        self.speed_error, self.turn_error = speed_error, turn_error
        self.wired = wired  # -1 = that motor's wires are swapped on the board
        self.min_clearance = math.inf  # closest the footprint came to a desk edge, cm
        self.clock_ms = 0

    def wheel(self, pwm, in1, in2, side):
        a, b = Pin.pins[in1].level, Pin.pins[in2].level
        direction = 1 if (a, b) == (1, 0) else -1 if (a, b) == (0, 1) else 0  # else brake
        duty = PWM.pwms[pwm].duty / 65535
        return direction * duty * FULL_CM_S * self.speed_error * self.wired[side]

    def advance(self, ms):
        assert Pin.pins[pc.PIN_STBY].level == 1, "TB6612 left in standby"
        left = self.wheel(pc.PIN_PWMA, pc.PIN_AIN1, pc.PIN_AIN2, 0)
        right = self.wheel(pc.PIN_PWMB, pc.PIN_BIN1, pc.PIN_BIN2, 1)
        dt = ms / 1000
        v, w = (left + right) / 2, (left - right) / TRACK_CM * self.turn_error
        self.th += w * dt  # left forward, right back: heading increases (y points down)
        self.x += v * math.cos(self.th) * dt
        self.y += v * math.sin(self.th) * dt
        self.clock_ms += ms
        self.min_clearance = min(self.min_clearance, self.x - ROBOT_RADIUS_CM,
                                 W_CM - ROBOT_RADIUS_CM - self.x, self.y - ROBOT_RADIUS_CM,
                                 H_CM - ROBOT_RADIUS_CM - self.y)


# ----- camera: the bot and markers drawn into the tilted camera image -----------

def scene(bot, target_cm=None):
    canvas = tc.top_down(objects=False, bot=False)
    if target_cm is not None:
        tc.paste_marker(canvas, shared.TARGET_ID, (target_cm[0] * 10, target_cm[1] * 10),
                        tc.CORNER_MM, 270)
    centre = (int(bot.x * 10) + tc.MARGIN, int(bot.y * 10) + tc.MARGIN)
    import cv2
    cv2.circle(canvas, centre, int(ROBOT_RADIUS_CM * 10), (60, 60, 60), -1)
    tc.paste_marker(canvas, shared.BOT_ID, (bot.x * 10, bot.y * 10), tc.BOT_MM,
                    math.degrees(bot.th))
    return tc.camera_view(canvas)


class Harness:
    """The link VisionHardware talks to: the camera worker on one side, the real
    Pico program (on the simulated bot's clock) on the other."""

    def __init__(self, worker, bot, target_cm=None):
        self.worker, self.bot, self.target_cm = worker, bot, target_cm
        self.replies, self.seq, self.next_id = [], 0, 1000
        self.vision_errors = []  # (position mm, heading deg) vision vs truth
        self.move_ms = []  # how long each Pico command took
        self.detect_s = []
        worker.link = self
        pc.send = self.replies.append
        pc.sleep_ms = bot.advance
        pc.ticks_ms = lambda: bot.clock_ms
        self.refresh()

    # PicoLink side used by the worker
    def update(self, desk_size, calibrated, bot, objects, target=None):
        self.state, self.target = (desk_size, calibrated, bot, objects), target

    def refresh(self):
        frame = scene(self.bot, self.target_cm)
        started = time.perf_counter()
        self.worker._detect(frame)
        self.detect_s.append(time.perf_counter() - started)
        seen = self.state[2]
        if seen is not None:
            err = math.dist((seen["x"], seen["y"]), (self.bot.x * 10, self.bot.y * 10))
            dh = (seen["hd"] - math.degrees(self.bot.th) + 180) % 360 - 180
            self.vision_errors.append((err, abs(dh)))

    # PicoLink side used by VisionHardware
    def latest(self):
        return self.state, 0.0

    def latest_target(self):
        return self.target, 0.0

    def send_command(self, do, a=None):
        self.next_id += 1
        self.seq += 1
        line = messaging.encode(self.seq, *self.state,
                                cmd={"id": self.next_id, "do": do, "a": a})
        message = pc.parse(line.decode())
        started = self.bot.clock_ms
        pc.on_update(message)  # runs the motors; the Pico prints DONE <id>
        self.move_ms.append(self.bot.clock_ms - started)
        self.refresh()  # the camera sees where the bot ended up
        return self.next_id

    def wait_done(self, cmd_id, timeout=5.0):
        return f"DONE {cmd_id}" in self.replies


WORKER = None


def setup(bot, target_cm=None):
    global WORKER
    if WORKER is None:
        WORKER = tc.learned_worker()
    pc.setup_motors()
    pc._last_cmd_id = None
    pc._speed[:] = [0.0, 0.0]
    pc.LEFT_REVERSED = pc.RIGHT_REVERSED = False
    link = Harness(WORKER, bot, target_cm)
    return link, VisionHardware(link, log=lambda text: None, settle=0.0)


# ----- checks ---------------------------------------------------------------------

def check_wiring():
    pc.setup_motors()
    pc.stop()
    used = {pc.PIN_PWMA, pc.PIN_AIN1, pc.PIN_AIN2, pc.PIN_BIN1, pc.PIN_BIN2, pc.PIN_PWMB,
            pc.PIN_STBY}
    check("TB6612 pins GP16-GP22 set up, none shared with the UART (GP0/GP1)",
          used == set(range(16, 23)) and not used & {pc.UART_TX_PIN, pc.UART_RX_PIN})
    check("PWM at 20 kHz, motors start stopped",
          all(PWM.pwms[p].hz == pc.PWM_FREQ and PWM.pwms[p].duty == 0
              for p in (pc.PIN_PWMA, pc.PIN_PWMB)))


def check_drive_check():
    bot = Bot(60, 30, 0)
    link, hw = setup(bot)
    lines = []
    problems = movement_check(hw, lines.append)
    check("--drive-check passes on a calibrated bot", problems == 0, lines[-1])
    errs = np.array(link.vision_errors)
    check("vision tracks the bot within 10 mm and 3 deg",
          errs[:, 0].max() < 10 and errs[:, 1].max() < 3,
          f"max {errs[:, 0].max():.1f} mm, {errs[:, 1].max():.1f} deg")
    longest = max(link.move_ms)
    check("each move finishes well inside the UART buffer and the 5 s DONE wait",
          longest < pc.MAX_MOVE_MS, f"longest move {longest} ms, cap {pc.MAX_MOVE_MS} ms")

    bot = Bot(60, 30, 0, speed_error=1.3, turn_error=0.75)
    _, hw = setup(bot)
    lines = []
    problems = movement_check(hw, lines.append)
    check("--drive-check flags a bot that needs calibrating", problems >= 2,
          f"{problems} problems")

    bot = Bot(60, 30, 0, wired=(1, -1))  # right motor wired backwards
    _, hw = setup(bot)
    lines = []
    flagged = movement_check(hw, lines.append)
    bot = Bot(60, 30, 0, wired=(1, -1))
    _, hw = setup(bot)
    pc.RIGHT_REVERSED = True  # the fix the README gives
    fixed = movement_check(hw, lambda text: None)
    pc.RIGHT_REVERSED = False
    check("a backwards motor is flagged, and RIGHT_REVERSED fixes it",
          flagged > 0 and fixed == 0, f"{flagged} problems -> {fixed}")


def check_goto():
    starts = [((25, 20, 0), (95, 40)), ((100, 45, 180), (20, 15)),
              ((60, 15, 90), (60, 50)), ((30, 45, 300), (110, 12)),
              ((90, 15, 135), (25, 48))]
    for label, speed, turn in (("calibrated", 1.0, 1.0), ("uncalibrated", 1.25, 0.8)):
        ok, moves, seconds, worst = 0, [], [], math.inf
        for (x, y, hd), target in starts:
            bot = Bot(x, y, hd, speed_error=speed, turn_error=turn)
            link, hw = setup(bot, target)
            polls = [0]

            def give_up():  # e.g. the target lost for good
                polls[0] += 1
                return polls[0] > 400
            result = goto_marker(hw, log=lambda text: None, follow=False,
                                 should_stop=give_up, max_moves=80, poll=0)
            distance = math.dist((bot.x, bot.y), target)
            ok += result["reason"] == "arrived" and distance <= ARRIVE_CM + 1
            moves.append(result["moves"])
            seconds.append(bot.clock_ms / 1000 + 0.3 * result["moves"])  # + Pi settle
            worst = min(worst, bot.min_clearance)
        check(f"--goto reaches the target marker from 5 starts ({label} motors)",
              ok == len(starts) and worst >= 0,
              f"{ok}/{len(starts)}, moves {moves}, about {max(seconds):.0f} s at most, "
              f"closest to an edge {worst:.1f} cm")


def check_policy():
    bot = Bot(60, 30, 20)
    link, hw = setup(bot)
    policy = DeskbotPolicy(str(ROOT / "sim" / "deskbot_policy.npz"))
    result = run(hw, policy, W_CM, H_CM, max_steps=60, start_heading=math.radians(20))
    check("--drive runs the policy 60 steps on the real Pico code without leaving the desk",
          result["steps"] == 60 and bot.min_clearance >= 0,
          f"coverage {result['coverage']:.0%}, closest to an edge {bot.min_clearance:.1f} cm, "
          f"{result['guard_interventions']} guard interventions")


def check_pico_receive():
    """The Pico's line reader: chunks split anywhere, a backlog, garbage."""
    def line(seq, cmd=None):
        return messaging.encode(seq, (1207, 597), True, {"x": 1, "y": 2, "hd": 3}, [],
                                cmd).decode()

    chunks, sent = [], []
    pc._reader = lambda: chunks.pop(0) if chunks else ""
    pc.send, pc._rx_buffer = sent.append, ""
    pc.ticks_ms = lambda: 0
    text = line(1)
    chunks[:] = [text[:10], text[10:]]
    first = pc._receive()
    chunks[:] = [line(2)[:-30]]
    partial = pc._receive()  # half a line: nothing yet, kept for later
    chunks[:] = [line(2)[-30:]]
    rest = pc._receive()
    check("Pico reads a line that arrives in pieces",
          [m["seq"] for m in first] == [1] and partial == [] and
          [m["seq"] for m in rest] == [2] and pc._rx_buffer == "")

    backlog = "".join(line(n, {"id": 7, "do": "move", "a": 0}) for n in range(3, 28))
    chunks[:] = [backlog[i:i + 64] for i in range(0, len(backlog), 64)]
    newest = pc._receive()
    check("after a move, the 25 queued lines give only the newest message",
          [m["seq"] for m in newest] == [27] and newest[0]["cmd"]["id"] == 7)

    chunks[:] = [line(28) + '{"seq": 29, "ca\n']
    fallback = pc._receive()
    check("a garbled newest line is reported and the one before it is used",
          [m["seq"] for m in fallback] == [28] and sent == ["ERR bad message"])

    chunks[:] = ["x" * 4100]
    pc._receive()
    check("a runaway line without a newline is dropped", pc._rx_buffer == "")
    pc._reader = None


def check_speed():
    link, _ = setup(Bot(60, 30, 0))
    for _ in range(20):
        link.refresh()
    ms = 1000 * float(np.median(link.detect_s[-20:]))
    check("vision keeps up: a 640x480 frame is processed in under 50 ms (this computer)",
          ms < 50, f"{ms:.1f} ms per frame, about {1000 / ms:.0f} fps")
    line = messaging.encode(1, *link.state)
    rate = len(line) * shared.DEFAULT_SEND_HZ
    check("link traffic fits 115200 baud with room to spare",
          rate < 0.25 * shared.SERIAL_BAUD / 10, f"{len(line)} bytes x {shared.DEFAULT_SEND_HZ}/s"
          f" = {rate} B/s of {shared.SERIAL_BAUD // 10}")


if __name__ == "__main__":
    check_wiring()
    check_pico_receive()
    check_drive_check()
    check_goto()
    check_policy()
    check_speed()
    print(f"\n{sum(results)}/{len(results)} checks passed")
    sys.exit(0 if all(results) else 1)
