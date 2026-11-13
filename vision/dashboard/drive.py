"""
The "drive" thread: runs the trained policy from sim/ (or the movement check,
or go-to-marker) on the coordinates the camera worker hands to the Pico link.

The loop is sim/pi_runner.run() with sim/vision_hardware.VisionHardware: the
camera gives the pose and the objects, the Pico carries out one move at a time.
"""

import sys
import threading
import time
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


def mode_of(options):
    """The mode name ("check", "goto" or "policy") these options ask for."""
    return "goto" if options.goto else "check" if options.check else "policy"


def list_policies():
    """Exported policies the dashboard may run: *.npz files directly in sim/."""
    return sorted(p.name for p in SIM_DIR.glob("*.npz"))


def policy_path(name):
    """The path of a policy the dashboard named. Only names from list_policies()
    are accepted, so the browser can never point the server at another file."""
    if not isinstance(name, str) or name not in list_policies():
        raise ValueError(f"unknown policy: {name}")
    return str(SIM_DIR / name)


def start_drive(link, options, log, stop_event):
    """Start the drive thread and return it. log(message, level) writes to the dashboard."""
    thread = threading.Thread(target=_drive, args=(link, options, log, stop_event),
                              daemon=True, name="drive")
    thread.start()
    return thread


def _drive(link, options, log, stop_event):
    """Run one drive mode to its end. Logs what happened and returns it as one line."""
    hw = None
    mode = mode_of(options)
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
            return f"{mode} ended (stopped before the bot and the desk were visible)"
        if options.goto:
            from goto_marker import goto_marker
            log("Drive: go-to-marker started: driving to the target marker (ID 5).", "ok")
            result = goto_marker(hw, lambda text: log(f"Drive: {text}"), stop_event.is_set)
            log(f"Drive: go-to-marker ended ({result['reason']}) after "
                f"{result['moves']} moves.")
            return f"goto ended ({result['reason']}) after {result['moves']} moves"
        if options.check:
            log("Drive: movement check started (forward, backward, turn +, turn -).")
            problems = movement_check(hw, lambda text: log(f"Drive: {text}"), stop_event.is_set)
            log(f"Drive: movement check finished, {problems} problem(s).",
                "ok" if problems == 0 else "warn")
            if stop_event.is_set():
                return "check ended (stopped)"
            return f"check finished, {problems} problem(s)"
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
        return (f"policy ended ({result['reason']}) after {result['steps']} steps, "
                f"coverage {result['coverage']:.0%}")
    except Exception as e:
        if stop_event.is_set():
            # Stop replaces the move the Pico was doing, so that move never reports
            # DONE and execute() raises. That is the stop working, not a fault.
            log(f"Drive: {mode} stopped.")
            return f"{mode} ended (stopped)"
        log(f"Drive: stopped by an error: {e}", "error")
        return f"{mode} stopped by an error: {e}"
    finally:
        if hw is not None:
            hw.stop()  # run() already stops; this covers the other ways out


class DriveManager:
    """Starts, stops and reports the single drive thread. One mode at a time."""

    def __init__(self, link, log):
        self.link, self.log = link, log
        self._lock = threading.Lock()
        self._thread = None
        self._stop = threading.Event()
        self.mode = None
        self.options = None
        self.started_at = None
        self.last_result = None  # e.g. "goto ended (stopped) after 12 moves"

    @property
    def running(self):
        return self._thread is not None and self._thread.is_alive()

    def start(self, options):
        """Start the mode in `options`. RuntimeError if one is already running."""
        with self._lock:
            if self.running:
                raise RuntimeError(f"'{self.mode}' is already running: stop it first")
            self._stop = threading.Event()
            self.options = options
            self.mode = mode_of(options)
            self.started_at = time.time()
            self.last_result = None
            self._thread = threading.Thread(
                target=self._run, args=(options, self._stop), daemon=True, name="drive")
            self._thread.start()

    def _run(self, options, stop_event):
        result = _drive(self.link, options, self.log, stop_event)
        with self._lock:
            if stop_event is self._stop:  # not a thread that outlived its stop()
                self.last_result = result

    def stop(self, timeout=8.0):
        """Stop the bot now, then end the drive thread. Safe to call any time.
        True when no drive thread is left running."""
        with self._lock:
            self._stop.set()
            thread = self._thread
        try:
            self.link.send_command("stop")  # don't wait for the thread to notice
        except Exception:  # the link may already be down; stopping must never fail
            pass
        if thread is not None:
            thread.join(timeout)
        return not self.running

    def status(self):
        with self._lock:
            running = self.running
            return {
                "running": running,
                "mode": self.mode if running else None,
                "started_at": self.started_at if running else None,
                "guard": None if self.options is None else self.options.guard,
                "last_result": self.last_result,
                "policies": list_policies(),
            }
