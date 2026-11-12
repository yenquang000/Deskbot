"""
The "drive" thread: runs the trained policy from sim/ (or the movement check,
or go-to-marker) on the coordinates the camera worker hands to the Pico link.

The loop is sim/pi_runner.run() with sim/vision_hardware.VisionHardware: the
camera gives the pose and the objects, the Pico carries out one move at a time.
"""

import sys
import threading
from dataclasses import dataclass
from pathlib import Path

SIM_DIR = Path(__file__).resolve().parent.parent.parent / "sim"


@dataclass
class DriveOptions:
    policy: str = None  # path to the exported .npz policy (--drive)
    check: bool = False  # run the movement check instead (--drive-check)
    guard: bool = True  # edge guard (--no-guard turns it off)
    clean_on_arrival: bool = False
    goto: bool = False  # drive to the target marker instead (--goto)


def start_drive(link, options, log, stop_event):
    """Start the drive thread and return it. log(message, level) writes to the dashboard."""
    thread = threading.Thread(target=_drive, args=(link, options, log, stop_event),
                              daemon=True, name="drive")
    thread.start()
    return thread


def _drive(link, options, log, stop_event):
    hw = None
    try:
        # sim/ uses flat imports (env, obs, baseline, ...), so its folder goes on
        # the path, after everything else so vision's own modules stay in front.
        if str(SIM_DIR) not in sys.path:
            sys.path.append(str(SIM_DIR))
        from pi_inference import DeskbotPolicy
        from pi_runner import run
        from vision_hardware import VisionHardware, movement_check

        policy = None if options.check or options.goto else DeskbotPolicy(options.policy)
        hw = VisionHardware(link, log=lambda text: log(f"Drive: {text}", "warn"))
        if options.goto:
            hw._desk_checked = True  # the desk-size warning is about the policy
        log("Drive: waiting for the 4 corner markers and the bot marker.")
        if not hw.wait_ready(stop_event.is_set):
            return
        if options.goto:
            from goto_marker import goto_marker
            log("Drive: go-to-marker started: driving to the target marker (ID 5).", "ok")
            result = goto_marker(hw, lambda text: log(f"Drive: {text}"), stop_event.is_set)
            log(f"Drive: go-to-marker ended ({result['reason']}) after "
                f"{result['moves']} moves.")
            return
        if options.check:
            log("Drive: movement check started (forward, backward, turn +, turn -).")
            problems = movement_check(hw, lambda text: log(f"Drive: {text}"), stop_event.is_set)
            log(f"Drive: movement check finished, {problems} problem(s).",
                "ok" if problems == 0 else "warn")
            return
        width, height = hw.desk_cm()
        log(f"Drive: policy started on a {width:.1f} x {height:.1f} cm desk, edge guard "
            f"{'on' if options.guard else 'OFF'}"
            f"{', clean on arrival' if options.clean_on_arrival else ''}.", "ok")
        result = run(hw, policy, width, height, guard=options.guard,
                     start_heading=hw.pose()[1] if hw.pose() else 0.0,
                     should_stop=stop_event.is_set,
                     clean_on_arrival=options.clean_on_arrival)
        log(f"Drive: ended ({result['reason']}) after {result['steps']} steps, "
            f"coverage {result['coverage']:.0%}, "
            f"{result['guard_interventions']} edge guard interventions.",
            "ok" if result["reason"] == "goal reached" else "warn")
    except Exception as e:
        log(f"Drive: stopped by an error: {e}", "error")
    finally:
        if hw is not None:
            hw.stop()  # run() already stops; this covers the other ways out
