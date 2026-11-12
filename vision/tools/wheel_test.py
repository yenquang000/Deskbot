"""
Wheel test over the Pi -> Pico link. Prop the bot up so the wheels are off the
ground, and stop the dashboard first: only one program can use the link.

  python tools/wheel_test.py              link check only, nothing moves
  python tools/wheel_test.py 0 1 2 3      moves, 2 s apart: 0 forward 5 cm,
                                          1 backward 5 cm, 2 turn +45, 3 turn -45
  python tools/wheel_test.py spin 10      both wheels forward for 10 s
  python tools/wheel_test.py spin -10     both wheels backward for 10 s

Ctrl+C during a spin stops the wheels at once. Exits with 0 when every command
was confirmed with DONE.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from hardware.messaging import PicoLink  # noqa: E402

MOVES = {0: "forward 5 cm", 1: "backward 5 cm",
         2: "turn +45 (left wheel forward, right backward)",
         3: "turn -45 (left wheel backward, right forward)"}


def parse_args(args):
    """The command line as a list of (do, a, description)."""
    steps = []
    i = 0
    while i < len(args):
        if args[i] == "spin":
            if i + 1 >= len(args):
                raise SystemExit("spin needs a number of seconds, e.g. spin 10")
            seconds = float(args[i + 1])
            if not 0 < abs(seconds) <= 30:
                raise SystemExit("spin takes 0 to 30 seconds (negative = backward)")
            way = "forward" if seconds > 0 else "backward"
            steps.append(("spin", int(seconds * 1000), f"both wheels {way} for {abs(seconds):g} s"))
            i += 2
        else:
            action = int(args[i])
            if action not in MOVES:
                raise SystemExit(f"unknown move {action}: use 0-3 or spin SECONDS")
            steps.append(("move", action, MOVES[action]))
            i += 1
    return steps


def main():
    steps = parse_args(sys.argv[1:])
    replies = []
    link = PicoLink("auto", 10, on_reply=replies.append)
    link.update((1207, 597), False, None, [])  # an uncalibrated frame: no bot, no objects
    link.start()

    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and not any(r.startswith("OK ") for r in replies):
        time.sleep(0.1)
    print("link:", link.status()["port"], "| last reply:", replies[-1] if replies else None)
    if not any(r.startswith("OK ") for r in replies):
        print("FAIL: no OK from the Pico within 5 s. Check its power, the TX/RX/GND wires,")
        print("      dtparam=uart0=on on the Pi, and that the dashboard isn't running.")
        link.stop()
        return 1

    all_done = True
    try:
        for n, (do, a, description) in enumerate(steps):
            if n:
                time.sleep(2)
            print(f"\n>>> {description}")
            started = time.monotonic()
            cmd_id = link.send_command(do, a)
            timeout = abs(a) / 1000 + 5 if do == "spin" else 5
            done = link.wait_done(cmd_id, timeout=timeout)
            all_done &= done
            print(f"    {'DONE' if done else 'NO DONE'} after {time.monotonic() - started:.1f} s")
    except KeyboardInterrupt:
        print("\nstopping the wheels")
        link.wait_done(link.send_command("stop"), timeout=2)
        all_done = False
    finally:
        time.sleep(0.3)
        link.stop()
        time.sleep(0.3)
    warnings = [r for r in replies if r.startswith(("WARN", "ERR"))]
    if warnings:
        print("\nPico warnings:", warnings[-3:])
    return 0 if all_done else 1


if __name__ == "__main__":
    sys.exit(main())
