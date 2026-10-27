"""
The browser dashboard: a small Flask server that streams the camera next to
live statistics, view options, settings and buttons for every keyboard command.

Open it at http://127.0.0.1:8000 (the program opens your browser for you).
"""

import threading
import webbrowser

from flask import Flask, Response, jsonify, render_template, request
from werkzeug.serving import make_server

import config
from hardware.messaging import PicoLink
from dashboard.worker import TOGGLES, CameraWorker


def create_app(worker, quit_event):
    app = Flask(__name__)

    def error(message, status=400):
        return jsonify({"ok": False, "error": message}), status

    @app.get("/")
    def index():
        return render_template("dashboard.html", toggles=TOGGLES)

    @app.get("/stream.mjpg")
    def stream():
        def frames():
            last_id = 0
            while not worker.stopped:
                new_id, jpeg = worker.wait_for_jpeg(last_id)
                if jpeg is None or new_id == last_id:
                    continue
                last_id = new_id
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n"
                       b"Content-Length: " + str(len(jpeg)).encode() + b"\r\n\r\n"
                       + jpeg + b"\r\n")

        return Response(frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

    @app.get("/api/status")
    def status():
        return jsonify(worker.status())

    @app.post("/api/action/<name>")
    def action(name):
        body = request.get_json(silent=True) or {}
        try:
            if name == "learn":
                if body.get("default"):
                    worker.start_learning(None)  # the default centred area
                else:
                    worker.start_learning((body["x"], body["y"], body["w"], body["h"]))
            elif name == "clear_desk":
                worker.clear_desk()
            elif name == "recalibrate":
                worker.recalibrate()
            elif name == "pause":
                worker.set_paused(True)
            elif name == "resume":
                worker.set_paused(False)
            elif name == "snapshot":
                return jsonify({"ok": True, "file": worker.snapshot()})
            elif name == "quit":
                worker.add_log("Quit requested from the dashboard.")
                quit_event.set()
            else:
                return error(f"Unknown action: {name}", 404)
        except KeyError:
            return error("Learning needs a desk box: x, y, w and h")
        except (TypeError, ValueError) as e:
            return error(str(e))
        return jsonify({"ok": True})

    @app.post("/api/view")
    def view():
        body = request.get_json(silent=True) or {}
        try:
            worker.set_toggles(body.get("toggles", {}))
        except ValueError as e:
            return error(str(e))
        return jsonify({"ok": True})

    @app.post("/api/settings")
    def settings():
        body = request.get_json(silent=True) or {}
        try:
            worker.update_settings(body, save=request.args.get("save") == "1")
        except ValueError as e:
            return error(str(e))
        return jsonify({"ok": True})

    return app


def run_dashboard(camera, host="127.0.0.1", port=8000, open_browser=True):
    """Run the camera worker and the dashboard until Quit is clicked or Ctrl+C."""
    settings = config.load()

    def on_reply(text):
        # Connection changes and Pico errors go in the log; routine replies only
        # show in the dashboard's Pico section, so they don't flood the log.
        if text.startswith("[link]"):
            worker.add_log(text[7:], "warn" if "Lost" in text else "ok")
        elif text.upper().startswith(("ERR", "WARN")):
            worker.add_log(f"Pico: {text}", "warn")

    link = PicoLink(settings.pico_port, settings.send_hz, on_reply)
    worker = CameraWorker(camera, settings, link)
    quit_event = threading.Event()
    app = create_app(worker, quit_event)

    try:
        server = make_server(host, port, app, threaded=True)
    except OSError as e:
        camera.release()
        print(f"[error] Could not start the dashboard on {host}:{port} ({e}). "
              f"Try another port with --port.")
        return 1

    worker.start()
    link.start()
    threading.Thread(target=server.serve_forever, daemon=True, name="dashboard").start()

    url = f"http://{'127.0.0.1' if host in ('0.0.0.0', '::') else host}:{port}"
    print(f"[ok] Using {camera.info()}")
    print(f"[ok] Dashboard running at {url}  (Ctrl+C or the Quit button to stop)")
    if host in ("0.0.0.0", "::"):
        print(f"[info] Reachable from other devices at http://<this machine's IP>:{port}")
    if open_browser:
        webbrowser.open(url)

    try:
        while not quit_event.wait(0.5):
            pass
    except KeyboardInterrupt:
        pass
    finally:
        worker.stop()
        link.stop()
        worker.join(timeout=3)
        link.join(timeout=2)
        server.shutdown()
        print("[ok] Dashboard stopped.")
    return 0
