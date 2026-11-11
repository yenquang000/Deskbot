"""
Deskbot Pico program (MicroPython). Receives desk coordinates and commands from
the Raspberry Pi over USB and carries the commands out, one at a time.

The Pi is the brain: it runs the trained policy and sends a "cmd" in its
messages. The Pico only moves by one action, runs the cleaner or stops, then
prints "DONE <id>" so the Pi knows the command is finished.

Install: copy this file to the Pico as main.py, so it runs on power-up:
  mpremote cp hardware/pico_controller.py :main.py
  (or open it in Thonny and "Save as" main.py on the Raspberry Pi Pico)

The Pi sends one JSON line roughly 10 times a second (format and coordinate
system: shared/config.py). Values are in millimetres, origin at corner marker 0,
x toward marker 1, y toward marker 3, heading 0 = facing +x, 90 = facing +y.

The motor and cleaner code goes in the TODO stubs below (drive_cm, turn_deg,
run_cleaner, stop).
"""

import json
import sys

try:
    import select
except ImportError:  # some MicroPython builds call it uselect
    import uselect as select

try:  # MicroPython millisecond timer
    from time import ticks_diff, ticks_ms
except ImportError:  # CPython, so parse() can be tested on a laptop
    import time

    def ticks_ms():
        return int(time.monotonic() * 1000)

    def ticks_diff(a, b):
        return a - b

LINK_TIMEOUT_MS = 500  # same as LINK_TIMEOUT_S in shared/config.py
STATUS_EVERY_MS = 2000  # how often to report back to the Pi

# One policy action. These must match Measurements in sim/env.py (base_step and
# turn_deg): the policy was trained with exactly these step sizes.
STEP_CM = 5
TURN_DEG = 45

_last_cmd_id = None  # id of the last command carried out


def parse(line):
    """
    Turn one received line into a dict, or None if it isn't a valid message.

    Result: {"seq", "t", "desk": [w, h], "cal": bool,
             "bot": {"x", "y", "hd"} or None,
             "objs": [{"id", "x", "y", "w", "h", "a"}, ...],
             "cmd": {"id", "do", "a"} or None}
    """
    try:
        msg = json.loads(line)
    except ValueError:
        return None
    if not isinstance(msg, dict):
        return None
    for key in ("seq", "desk", "cal", "objs"):
        if key not in msg:
            return None
    msg.setdefault("bot", None)
    msg.setdefault("cmd", None)
    return msg


# ----- commands from the Pi -----------------------------------------------------

def on_update(state):
    """
    Called with every new message. Carries out state["cmd"] if it is new.

    The Pi repeats a command in every message until it sees DONE, so a command
    whose id equals the last one carried out is ignored.
    """
    global _last_cmd_id
    cmd = state["cmd"]
    if not cmd or cmd.get("id") is None or cmd["id"] == _last_cmd_id:
        return
    _last_cmd_id = cmd["id"]
    do = cmd.get("do")
    if do == "move":
        run_action(cmd.get("a"))
    elif do == "clean":
        run_cleaner(cmd.get("a") or 0)
    else:  # "stop", and anything unknown: stopping is the safe answer
        stop()
    print("DONE %d" % cmd["id"])


def run_action(action):
    """One policy action: 0 forward, 1 backward, 2 heading +45, 3 heading -45."""
    if action == 0:
        drive_cm(STEP_CM)
    elif action == 1:
        drive_cm(-STEP_CM)
    elif action == 2:
        turn_deg(TURN_DEG)
    elif action == 3:
        turn_deg(-TURN_DEG)
    stop()


def on_lost():
    """Called when no message has arrived for LINK_TIMEOUT_MS: stop safely."""
    stop()


# ----- motors and cleaner: TODO, fill these in for the real hardware -----------
# These may block until the move is finished. While they block, the Pi's
# messages queue up on USB, so keep each one short (about a second or two).

def drive_cm(cm):
    """TODO: drive straight by cm (negative = backward), then return."""
    pass


def turn_deg(deg):
    """
    TODO: turn on the spot by deg, then return.

    Direction is defined by the desk heading, never by "the robot's left":
    a positive deg must make the camera's heading (hd) INCREASE. y points
    toward marker 3, usually down in the image, so positive is clockwise when
    seen from a camera looking down at the desk. Check it with
    `python main.py --drive-check`, which flags a turn wired backwards.
    """
    pass


def run_cleaner(ms):
    """TODO: run the cleaner for ms milliseconds, then return."""
    pass


def stop():
    """TODO: stop the motors."""
    pass


# ----- receiving loop ----------------------------------------------------------

def main():
    poller = select.poll()
    poller.register(sys.stdin, select.POLLIN)
    buffer = ""
    state = None
    last_message = ticks_ms()
    last_status = ticks_ms()
    lost = True
    print("Deskbot Pico ready")

    while True:
        # Read whatever has arrived without waiting, one character at a time,
        # and handle each complete line. A command can block in on_update();
        # all the lines queued meanwhile are read here before the link timeout
        # is checked below, so keep this order.
        while poller.poll(0):
            char = sys.stdin.read(1)
            if char == "\n":
                message = parse(buffer)
                buffer = ""
                if message is None:
                    print("ERR bad message")
                    continue
                state = message
                last_message = ticks_ms()
                lost = False
                on_update(state)
            elif len(buffer) < 4000:
                buffer += char
            else:
                buffer = ""  # a line that long is garbage; start over

        now = ticks_ms()
        if not lost and ticks_diff(now, last_message) > LINK_TIMEOUT_MS:
            lost = True
            print("WARN no messages from the Pi, stopping")
            on_lost()

        if ticks_diff(now, last_status) > STATUS_EVERY_MS:
            last_status = now
            if state is not None and not lost:
                bot = state["bot"]
                where = "bot ?" if bot is None else "bot %d,%d %ddeg" % (
                    bot["x"], bot["y"], bot["hd"])
                print("OK seq=%d cal=%d %s objs=%d" % (
                    state["seq"], state["cal"], where, len(state["objs"])))


if __name__ == "__main__":
    main()
