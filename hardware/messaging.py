"""
Pi side of the Pi -> Pico link: builds the coordinate messages and sends them
over USB serial from a background thread. The message format and coordinate
system are described in shared/config.py; the Pico side is pico_controller.py.
"""

import glob
import json
import threading
import time
from collections import deque

import serial

from shared import config as shared

_START = time.monotonic()


def encode(seq, desk_size, calibrated, bot, objects):
    """
    Build one message line (bytes ending in a newline).

    desk_size: (width_mm, height_mm). bot: {"x", "y", "hd"} or None.
    objects: list of {"id", "x", "y", "w", "h", "a"}. Numbers are rounded to ints.
    """
    message = {
        "seq": seq,
        "t": int((time.monotonic() - _START) * 1000),
        "desk": [round(desk_size[0]), round(desk_size[1])],
        "cal": bool(calibrated),
        "bot": None,
        "objs": [],
    }
    if calibrated:
        if bot is not None:
            message["bot"] = {"x": round(bot["x"]), "y": round(bot["y"]),
                              "hd": round(bot["hd"]) % 360}
        message["objs"] = [{k: round(o[k]) for k in ("id", "x", "y", "w", "h", "a")}
                           for o in objects]
    return (json.dumps(message, separators=(",", ":")) + "\n").encode()


def find_pico_port():
    """The first USB serial port that looks like a Pico, or None."""
    ports = sorted(glob.glob("/dev/ttyACM*")) + sorted(glob.glob("/dev/cu.usbmodem*"))
    if ports:
        return ports[0]
    try:  # Windows and anything glob misses: ask pyserial (Raspberry Pi vendor id 0x2E8A)
        from serial.tools import list_ports
        for port in list_ports.comports():
            if port.vid == 0x2E8A:
                return port.device
    except ImportError:
        pass
    return None


class PicoLink(threading.Thread):
    """
    Sends the latest coordinates to the Pico at a fixed rate and collects what
    the Pico prints back. Reconnects by itself if the Pico is unplugged.
    """

    RETRY_SECONDS = 2.0

    def __init__(self, port="auto", send_hz=shared.DEFAULT_SEND_HZ, on_reply=None):
        super().__init__(daemon=True, name="pico-link")
        self.port_setting = port
        self.send_hz = send_hz
        self.on_reply = on_reply  # called with each line the Pico prints
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._state = None  # (desk_size, calibrated, bot, objects)
        self._serial = None
        self._buffer = b""

        self.port = None
        self.seq = 0
        self.sent = 0
        self.errors = 0
        self.last_message = None
        self.last_reply = None
        self.replies = deque(maxlen=20)
        self.last_error = None

    def configure(self, port, send_hz):
        """Apply new settings; a changed port reconnects."""
        with self._lock:
            if port != self.port_setting:
                self.port_setting = port
                self._close()
            self.send_hz = send_hz

    def update(self, desk_size, calibrated, bot, objects):
        """Hand over the newest coordinates; only the latest is ever sent."""
        with self._lock:
            self._state = (desk_size, calibrated, bot, objects)

    def stop(self):
        self._stop_event.set()

    @property
    def connected(self):
        return self._serial is not None

    def status(self):
        with self._lock:
            return {
                "connected": self.connected,
                "port": self.port,
                "port_setting": self.port_setting,
                "sent": self.sent,
                "send_hz": self.send_hz,
                "last_message": self.last_message,
                "last_reply": self.last_reply,
                "error": None if self.connected else self.last_error,
            }

    # ----- background thread ----------------------------------------------

    def run(self):
        next_retry = 0.0
        try:
            while not self._stop_event.is_set():
                started = time.monotonic()
                if self._serial is None and started >= next_retry:
                    if not self._open():
                        next_retry = started + self.RETRY_SECONDS
                if self._serial is not None:
                    self._send_latest()
                    self._read_replies()
                self._stop_event.wait(max(0.0, 1.0 / self.send_hz - (time.monotonic() - started)))
        finally:
            with self._lock:
                self._close()

    def _open(self):
        port = find_pico_port() if self.port_setting == "auto" else self.port_setting
        if port is None:
            self.last_error = "No Pico found (is it plugged in by USB?)"
            return False
        try:
            # serial_for_url also accepts test URLs such as "loop://".
            self._serial = serial.serial_for_url(port, baudrate=shared.SERIAL_BAUD,
                                                 timeout=0, write_timeout=0.2)
        except (serial.SerialException, OSError) as e:
            self.last_error = f"Could not open {port}: {e}"
            return False
        self.port = port
        self.last_error = None
        self._buffer = b""
        self._reply(f"[link] connected on {port}")
        return True

    def _close(self):
        if self._serial is not None:
            try:
                self._serial.close()
            except (serial.SerialException, OSError):
                pass
        self._serial = None

    def _send_latest(self):
        with self._lock:
            state = self._state
        if state is None:
            return
        self.seq += 1
        line = encode(self.seq, *state)
        try:
            self._serial.write(line)
        except (serial.SerialException, OSError) as e:
            self._lost(e)
            return
        self.sent += 1
        self.last_message = line.decode().strip()

    def _read_replies(self):
        if self._serial is None:
            return
        try:
            waiting = self._serial.in_waiting
            if waiting:
                self._buffer += self._serial.read(waiting)
        except (serial.SerialException, OSError) as e:
            self._lost(e)
            return
        while b"\n" in self._buffer:
            line, self._buffer = self._buffer.split(b"\n", 1)
            text = line.decode(errors="replace").strip()
            if text:
                self._reply(text)
        self._buffer = self._buffer[-4096:]  # never grow without bound

    def _reply(self, text):
        self.last_reply = text
        self.replies.append(text)
        if self.on_reply:
            self.on_reply(text)

    def _lost(self, error):
        self.errors += 1
        self.last_error = f"Lost connection to {self.port}: {error}"
        self._reply(f"[link] {self.last_error}")
        with self._lock:
            self._close()
