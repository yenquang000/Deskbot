"""
Deskbot Pico program (MicroPython). Receives desk coordinates from the
Raspberry Pi over USB and decides how the bot moves.

Install: copy this file to the Pico as main.py, so it runs on power-up:
  mpremote cp hardware/pico_controller.py :main.py
  (or open it in Thonny and "Save as" main.py on the Raspberry Pi Pico)

The Pi sends one JSON line roughly 10 times a second (format and coordinate
system: shared/config.py). Values are in millimetres, origin at corner marker 0,
x toward marker 1, y toward marker 3, heading 0 = facing +x, 90 = facing +y.

All movement calculations belong in on_update() below.
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


def parse(line):
    """
    Turn one received line into a dict, or None if it isn't a valid message.

    Result: {"seq", "t", "desk": [w, h], "cal": bool,
             "bot": {"x", "y", "hd"} or None,
             "objs": [{"id", "x", "y", "w", "h", "a"}, ...]}
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
    return msg


# ----- your bot logic goes here ------------------------------------------------

def on_update(state):
    """
    Called with every new message. Do the movement calculations here.

    state["cal"] is False until the Pi has found all 4 corner markers; then
    there are no coordinates, so the bot should wait.
    state["bot"] is None when the bot's marker is not visible.
    """
    if not state["cal"] or state["bot"] is None:
        stop()
        return
    bot = state["bot"]
    objs = state["objs"]
    # Example only: the nearest object, as a target to drive to.
    if objs:
        target = min(objs, key=lambda o: (o["x"] - bot["x"]) ** 2 + (o["y"] - bot["y"]) ** 2)
        _ = target  # TODO: steer toward target["x"], target["y"] from bot x, y, heading
    else:
        stop()


def on_lost():
    """Called when no message has arrived for LINK_TIMEOUT_MS: stop safely."""
    stop()


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
        # and handle each complete line.
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
