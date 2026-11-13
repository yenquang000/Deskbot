"""
Checks for the dashboard's drive controls: DriveManager and the
/api/drive/start, /api/drive/stop and /api/status routes.

Uses Flask's test client and the fake link from sim/test_vision_bridge.py (a
simulated bot), so no camera, Pico or browser is needed.

Usage (from the vision folder):
  python tools/test_drive_api.py
"""

import json
import socket
import sys
import threading
import time
import urllib.request
from pathlib import Path

import numpy as np

VISION_DIR = Path(__file__).resolve().parent.parent
SIM_DIR = VISION_DIR.parent / "sim"
sys.path.insert(0, str(VISION_DIR.parent))
sys.path.insert(0, str(VISION_DIR))
sys.path.append(str(SIM_DIR))

import config  # noqa: E402
from dashboard.drive import DriveManager, DriveOptions, list_policies, policy_path  # noqa: E402
from dashboard.server import create_app, run_dashboard  # noqa: E402
from dashboard.worker import CameraWorker  # noqa: E402
from env import DeskbotEnv  # noqa: E402
from test_vision_bridge import FakeLink  # noqa: E402

results = []


def check(name, passed, detail=""):
    results.append(bool(passed))
    print(f"{'PASS' if passed else 'FAIL'}  {name}{f'  ({detail})' if detail else ''}")


class BlankCamera:
    """A fake camera that shows an empty grey frame."""

    def read(self):
        return np.full((480, 640, 3), 128, np.uint8)

    def info(self):
        return "Blank test camera 640x480"

    def release(self):
        pass


class SlowLink(FakeLink):
    """A link whose moves never finish by themselves: wait_done blocks until
    another command (the dashboard's stop) replaces the move, like PicoLink."""

    def __init__(self, env):
        super().__init__(env)
        self._id_lock = threading.Lock()

    def send_command(self, do, a=None):
        with self._id_lock:  # ids in order, as PicoLink gives them
            self.commands.append((do, a))
            return len(self.commands)

    def wait_done(self, cmd_id, timeout=5.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if len(self.commands) != cmd_id:  # replaced by a newer command
                return False
            time.sleep(0.01)
        return False


def make(link_class=FakeLink):
    """A test client on a fresh app: (client, manager, link, worker, quit_event)."""
    env = DeskbotEnv(stage=3, randomize=False)
    env.reset(seed=0)
    link = link_class(env)
    worker = CameraWorker(BlankCamera(), config.Settings())  # never started
    manager = DriveManager(link, worker.add_log)
    quit_event = threading.Event()
    app = create_app(worker, quit_event, manager)
    return app.test_client(), manager, link, worker, quit_event


def drive_status(client):
    return client.get("/api/status").get_json()["drive"]


def wait_until(condition, seconds=8.0):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.02)
    return condition()


def check_start_and_status():
    client, manager, link, worker, _ = make()
    idle = drive_status(client)
    check("status has an idle drive entry before anything starts",
          idle["running"] is False and idle["mode"] is None and idle["last_result"] is None
          and idle["guard"] is None, idle)
    check("status lists the policies in sim/",
          idle["policies"] == list_policies() and "deskbot_policy.npz" in idle["policies"],
          idle["policies"])

    res = client.post("/api/drive/start", json={"mode": "goto"})
    status = drive_status(client)
    check("start goto answers 200 and status shows it running",
          res.status_code == 200 and res.get_json() == {"ok": True}
          and status["running"] is True and status["mode"] == "goto"
          and status["started_at"] is not None, (res.status_code, status))
    check("the start is written to the dashboard log",
          any("'goto' started from the dashboard" in e["message"] for e in worker.log))

    res = client.post("/api/drive/start", json={"mode": "check"})
    check("a second start while running answers 409",
          res.status_code == 409 and res.get_json()["ok"] is False
          and drive_status(client)["mode"] == "goto", (res.status_code, res.get_json()))

    res = client.post("/api/drive/stop")
    stopped = wait_until(lambda: not drive_status(client)["running"])
    status = drive_status(client)
    check("stop sends a stop command to the Pico",
          res.status_code == 200 and ("stop", None) in link.commands, link.commands)
    check("stop ends the drive thread within 8 s",
          stopped and res.get_json() == {"ok": True, "stopped": True} and status["mode"] is None,
          (res.get_json(), status))
    check("the last result says how the mode ended",
          status["last_result"] is not None and "goto" in status["last_result"]
          and "stopped" in status["last_result"], status["last_result"])


def check_bad_requests():
    client, manager, link, _, _ = make()
    for body, name in (({"mode": "fly"}, "an unknown mode"),
                       ({}, "a missing mode"),
                       ({"mode": "policy", "policy": "../../etc/passwd"}, "a policy path"),
                       ({"mode": "policy", "policy": "/etc/passwd"}, "an absolute policy path"),
                       ({"mode": "policy", "policy": "no_such.npz"}, "a policy not in sim/"),
                       ({"mode": "policy", "policy": ["deskbot_policy.npz"]}, "a non-text policy")):
        res = client.post("/api/drive/start", json=body)
        check(f"{name} answers 400 and starts nothing",
              res.status_code == 400 and res.get_json()["ok"] is False and not manager.running,
              (res.status_code, res.get_json()))
    check("nothing was sent to the Pico by the refused starts", link.commands == [], link.commands)

    try:
        policy_path("../sim/deskbot_policy.npz")
        check("policy_path refuses a name with a folder in it", False)
    except ValueError:
        check("policy_path refuses a name with a folder in it", True)
    check("policy_path gives the file in sim/ for a listed name",
          Path(policy_path("deskbot_policy.npz")) == SIM_DIR / "deskbot_policy.npz")


def check_restartable():
    client, manager, link, _, _ = make()
    first = client.post("/api/drive/start", json={"mode": "goto"}).status_code
    client.post("/api/drive/stop")
    wait_until(lambda: not manager.running)
    link.answer = False  # moves never finish, so the policy stays running until stopped
    second = client.post("/api/drive/start",
                         json={"mode": "policy", "policy": "deskbot_policy.npz",
                               "guard": False, "clean_on_arrival": True})
    status = drive_status(client)
    check("start, stop, then a different mode starts",
          first == 200 and second.status_code == 200 and status["running"]
          and status["mode"] == "policy", (first, second.status_code, status))
    check("the policy options reach the drive thread",
          status["guard"] is False and manager.options.clean_on_arrival is True
          and manager.options.policy == policy_path("deskbot_policy.npz"), manager.options)
    client.post("/api/drive/stop")
    check("the second mode stops too", wait_until(lambda: not manager.running))

    res = client.post("/api/drive/stop")
    check("stop while idle is harmless: 200 and one more stop to the Pico",
          res.status_code == 200 and res.get_json()["stopped"] is True
          and link.commands[-1] == ("stop", None), (res.get_json(), link.commands[-3:]))


def check_stop_mid_move():
    client, manager, link, _, _ = make(SlowLink)
    link.target_cm = (100, 30)  # a target to drive to, so a move starts
    client.post("/api/drive/start", json={"mode": "goto"})
    moving = wait_until(lambda: any(c[0] == "move" for c in link.commands))
    began = time.monotonic()
    res = client.post("/api/drive/stop")
    took = time.monotonic() - began
    result = drive_status(client)["last_result"]
    check("stop during a blocking move ends the thread without waiting for the move",
          moving and res.get_json()["stopped"] is True and took < 3.0, f"{took:.1f} s")
    check("a move cut short by stop is reported as stopped, not as an error",
          result is not None and "stopped" in result and "error" not in result, result)


def check_results():
    client, manager, link, _, _ = make()
    manager.start(DriveOptions(policy="no_such_policy.npz"))
    wait_until(lambda: not manager.running)
    result = manager.status()["last_result"]
    check("a failed start (bad policy file) leaves an error result",
          result is not None and "error" in result, result)


def check_quit():
    client, manager, link, _, quit_event = make()
    client.post("/api/drive/start", json={"mode": "goto"})
    res = client.post("/api/action/quit")
    check("Quit stops the drive and sets the quit event",
          res.status_code == 200 and quit_event.is_set()
          and wait_until(lambda: not manager.running) and ("stop", None) in link.commands,
          link.commands)


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def http(port, path, post=False):
    request = urllib.request.Request(f"http://127.0.0.1:{port}{path}",
                                     data=b"{}" if post else None,
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=5) as response:
        return json.loads(response.read())


def check_cli_path():
    """run_dashboard() itself: with DriveOptions it starts that mode, without it idles."""
    for options, want in ((DriveOptions(goto=True), "goto"), (None, None)):
        port, out = free_port(), []
        thread = threading.Thread(
            target=lambda: out.append(run_dashboard(BlankCamera(), "127.0.0.1", port,
                                                    open_browser=False, drive=options)),
            daemon=True)
        thread.start()
        status = None

        def ready():
            nonlocal status
            try:
                status = http(port, "/api/status")["drive"]
            except OSError:
                return False
            return status["mode"] == want
        up = wait_until(ready)
        label = f"run_dashboard(drive={'DriveOptions(goto=True)' if options else 'None'})"
        check(f"{label} starts the manager {'in goto' if want else 'idle'}",
              up and status["running"] is (want is not None), status)
        if up:
            http(port, "/api/action/quit", post=True)
        thread.join(15)
        check(f"{label} shuts down cleanly on Quit", not thread.is_alive() and out == [0], out)


def main():
    check_start_and_status()
    check_bad_requests()
    check_restartable()
    check_stop_mid_move()
    check_results()
    check_quit()
    check_cli_path()
    print(f"\n{sum(results)}/{len(results)} checks passed")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
