"""
Deskbot Pico program (MicroPython). Receives desk coordinates and commands from
the Raspberry Pi over GPIO serial (UART) or USB and carries the commands out,
one at a time. Choose the link with LINK below.

UART wiring (LINK = "uart", the default), 3.3 V logic on both boards:
  Pico GP0 (UART0 TX, pin 1)  ->  Pi GPIO15 (RXD, pin 10)
  Pico GP1 (UART0 RX, pin 2)  <-  Pi GPIO14 (TXD, pin 8)
  Pico GND (pin 3)            --  Pi GND (pin 6)
The Pi needs `dtparam=uart0=on` in /boot/firmware/config.txt and no serial
console on the port (raspi-config -> Interface Options -> Serial Port: login
shell No, hardware Yes), then a reboot;
the port is /dev/ttyAMA0, which messaging.py finds by itself.

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
    from time import sleep_ms, ticks_diff, ticks_ms
except ImportError:  # CPython, so parse() can be tested on a laptop
    import time

    def ticks_ms():
        return int(time.monotonic() * 1000)

    def ticks_diff(a, b):
        return a - b

    def sleep_ms(ms):
        time.sleep(ms / 1000)

# How the Pi is connected: "uart" = GPIO pins below, "usb" = USB cable.
LINK = "uart"
UART_ID = 0  # UART0
UART_TX_PIN = 0  # GP0 -> Pi GPIO15 (RXD)
UART_RX_PIN = 1  # GP1 <- Pi GPIO14 (TXD)
UART_BAUD = 115200  # same as SERIAL_BAUD in shared/config.py
# Room for the messages that queue up while a move blocks (about 300 bytes each,
# 10 a second); the default buffer is too small and would garble them.
UART_RXBUF = 8192

LINK_TIMEOUT_MS = 500  # same as LINK_TIMEOUT_S in shared/config.py
STATUS_EVERY_MS = 2000  # how often to report back to the Pi

# One policy action. These must match Measurements in sim/env.py (base_step and
# turn_deg): the policy was trained with exactly these step sizes.
STEP_CM = 5
TURN_DEG = 45

# ----- motor wiring (Deskbot Drive Test sheet, rev D1) --------------------------
# TB6612FNG channel A = left motor, channel B = right motor. VM is the 7.4 V
# battery, VCC is the Pico's 3V3 OUT. Encoders are not connected yet, so moves
# are timed (see CM_PER_S and DEG_PER_S below).
PIN_PWMA = 16  # left speed
PIN_AIN1 = 17  # left direction
PIN_AIN2 = 18
PIN_BIN1 = 19  # right direction
PIN_BIN2 = 20
PIN_PWMB = 21  # right speed
PIN_STBY = 22  # 1 = driver on
PWM_FREQ = 20000  # above hearing, so the motors don't whine

# Flip these instead of rewiring if a wheel turns the wrong way. Check with the
# wheels off the ground: "move 0" (forward) must turn both wheels forward.
LEFT_REVERSED = False
RIGHT_REVERSED = False

# Duty cycles (0..1). Kept low for the desk; the drive-test sheet caps at 0.6.
DRIVE_SPEED = 0.5
TURN_SPEED = 0.4
# Max change in duty per 5 ms step: 0 to full in about 0.25 s. A slow start keeps
# the start-up current under the battery pack's 4 A cutoff, which would
# otherwise cut power to the Pi as well.
RAMP_PER_STEP = 0.02
STEP_MS = 5

# Timed-move calibration: how far the bot goes per second at DRIVE_SPEED, and
# how fast it turns at TURN_SPEED. These are estimates (N20 200 rpm at 12 V runs
# about 120 rpm on 7.4 V; 34 mm wheels). Calibrate with the camera:
#   cd vision && python main.py --drive-check
# The log shows the measured step and turn. New value = old value x measured /
# expected, e.g. a forward step measured 6.2 cm instead of 5 cm:
# CM_PER_S = 9.0 x 6.2 / 5 = 11.2. Then upload this file to the Pico again.
CM_PER_S = 9.0
DEG_PER_S = 70.0
# Safety cap per move or clean. While one blocks, the Pi's messages (about
# 3 KB/s) queue in UART_RXBUF, which holds about 2.7 s of them; a longer block
# would overflow it and garble the queued lines. Keep this under that (and under
# the Pi's 5 s wait for DONE), or raise UART_RXBUF with it.
MAX_MOVE_MS = 2500
# "spin" test command: both wheels forward at DRIVE_SPEED for a = ms
# (negative = backward). Unlike a move it keeps reading the link while it runs.
SPIN_MAX_MS = 30000

# Cleaner (vacuum MOSFET module). None = not wired yet: a clean command just
# waits for its duration so the timing on the Pi stays the same.
PIN_CLEANER = None

_last_cmd_id = None  # id of the last command carried out
_motors = None  # set by setup_motors() on the Pico; None in CPython tests
_cleaner = None
_speed = [0.0, 0.0]  # current duty of the left and right motor, -1..1
_uart = None  # machine.UART once main() opens it (LINK = "uart")
_reader = None  # read_available() from open_link(); None in CPython tests
_rx_buffer = ""  # the incomplete line received so far
_last_rx = 0  # ticks_ms() of the last valid message
_pending = None  # a message with a new command that arrived during a spin


def send(text):
    """
    Send one line to the Pi. print() always goes to USB as well, so the
    Pico's output can still be watched in Thonny or `mpremote` while debugging.
    """
    print(text)
    if _uart is not None:
        _uart.write(text + "\n")


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
    elif do == "spin":
        spin_wheels(cmd.get("a") or 0, cmd["id"])
    else:  # "stop", and anything unknown: stopping is the safe answer
        stop()
    send("DONE %d" % cmd["id"])


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


# ----- motors and cleaner --------------------------------------------------------
# These block until the move is finished. While they block, the Pi's messages
# queue up on the link, so each one stays short (about a second).

def setup_motors():
    """Set up the TB6612 pins with the motors stopped. Called once by main()."""
    global _motors, _cleaner
    from machine import PWM, Pin
    pwma, pwmb = PWM(Pin(PIN_PWMA)), PWM(Pin(PIN_PWMB))
    for pwm in (pwma, pwmb):
        pwm.freq(PWM_FREQ)
        pwm.duty_u16(0)
    _motors = (
        (pwma, Pin(PIN_AIN1, Pin.OUT, value=0), Pin(PIN_AIN2, Pin.OUT, value=0)),
        (pwmb, Pin(PIN_BIN1, Pin.OUT, value=0), Pin(PIN_BIN2, Pin.OUT, value=0)),
    )
    Pin(PIN_STBY, Pin.OUT, value=1)  # driver on; the motors are still at 0
    if PIN_CLEANER is not None:
        _cleaner = Pin(PIN_CLEANER, Pin.OUT, value=0)


def _set_motor(side, duty):
    """Set one motor (0 = left, 1 = right) to duty -1..1 (negative = backward)."""
    duty = max(-1.0, min(1.0, duty))
    _speed[side] = duty
    if _motors is None:  # CPython tests: no hardware
        return
    pwm, in1, in2 = _motors[side]
    if (LEFT_REVERSED, RIGHT_REVERSED)[side]:
        duty = -duty
    in1.value(1 if duty > 0 else 0)
    in2.value(1 if duty < 0 else 0)
    pwm.duty_u16(int(abs(duty) * 65535))


def _ramp_to(left, right):
    """Step both motors toward the target duties at RAMP_PER_STEP per step."""
    while _speed[0] != left or _speed[1] != right:
        for side, target in ((0, left), (1, right)):
            now = _speed[side]
            step = max(-RAMP_PER_STEP, min(RAMP_PER_STEP, target - now))
            _set_motor(side, target if abs(target - now) <= RAMP_PER_STEP else now + step)
        sleep_ms(STEP_MS)


def _timed_move(left_dir, right_dir, duty, amount, rate):
    """
    Run the wheels in the given directions (+1/-1) until about `amount` (cm or
    degrees) has been covered at `rate` per second at full `duty`. Distance
    covered while ramping up and down is counted, so short moves still come
    out the right length.
    """
    done = 0.0
    ramp_steps = duty / RAMP_PER_STEP
    # What the ramp down will add after we start slowing down (a triangle).
    ramp_down = rate * (ramp_steps * STEP_MS / 1000) / 2
    started = ticks_ms()
    level = 0.0  # 0..duty, the same for both wheels
    while done < amount - ramp_down:
        if ticks_diff(ticks_ms(), started) > MAX_MOVE_MS:
            break  # never run away: the Pi stops waiting at 5 s anyway
        level = min(duty, level + RAMP_PER_STEP)
        _set_motor(0, left_dir * level)
        _set_motor(1, right_dir * level)
        sleep_ms(STEP_MS)
        done += rate * (level / duty) * STEP_MS / 1000
    _ramp_to(0.0, 0.0)


def drive_cm(cm):
    """Drive straight by cm (negative = backward), then return."""
    direction = 1 if cm >= 0 else -1
    _timed_move(direction, direction, DRIVE_SPEED, abs(cm), CM_PER_S)


def turn_deg(deg):
    """
    Turn on the spot by deg, then return.

    Direction is defined by the desk heading, never by "the robot's left":
    a positive deg must make the camera's heading (hd) INCREASE. y points
    toward marker 3, usually down in the image, so positive is clockwise when
    seen from a camera looking down at the desk: left wheel forward, right
    wheel backward. `python main.py --drive-check` flags a turn wired
    backwards. If forward is right but turns are backwards, left and right
    are swapped: swap the TB6612's AO and BO wire pairs.
    """
    direction = 1 if deg >= 0 else -1
    _timed_move(direction, -direction, TURN_SPEED, abs(deg), DEG_PER_S)


def spin_wheels(ms, cmd_id=None):
    """
    Test command: run both wheels forward (negative ms = backward) at
    DRIVE_SPEED for ms milliseconds, capped at SPIN_MAX_MS.

    A spin lasts too long to block like a move, so it keeps reading the link:
    it stops early if the Pi goes quiet for LINK_TIMEOUT_MS, or if a message
    brings a different command (that message is then handled next).
    """
    global _pending
    direction = 1 if ms >= 0 else -1
    ms = min(abs(int(ms)), SPIN_MAX_MS)
    started = ticks_ms()
    level = 0.0
    while ticks_diff(ticks_ms(), started) < ms:
        level = min(DRIVE_SPEED, level + RAMP_PER_STEP)
        _set_motor(0, direction * level)
        _set_motor(1, direction * level)
        sleep_ms(STEP_MS)
        if _reader is None:  # CPython tests: no link to watch
            continue
        new_command = None
        for message in _receive():
            cmd = message["cmd"]
            if cmd and cmd.get("id") is not None and cmd["id"] != cmd_id:
                new_command = message
        if new_command is not None:
            _pending = new_command
            break
        if ticks_diff(ticks_ms(), _last_rx) > LINK_TIMEOUT_MS:
            send("WARN no messages from the Pi, stopping the spin")
            break
    _ramp_to(0.0, 0.0)
    stop()


def run_cleaner(ms):
    """Run the cleaner for ms milliseconds (or just wait if it isn't wired)."""
    ms = max(0, min(int(ms), MAX_MOVE_MS))
    if _cleaner is not None:
        _cleaner.value(1)
    try:
        sleep_ms(ms)
    finally:
        if _cleaner is not None:
            _cleaner.value(0)


def stop():
    """Stop both motors at once (short brake), and the cleaner."""
    _speed[0] = _speed[1] = 0.0
    if _cleaner is not None:
        _cleaner.value(0)
    if _motors is None:
        return
    for pwm, in1, in2 in _motors:
        pwm.duty_u16(0)
        in1.value(1)  # IN1 = IN2 = 1 is the TB6612's short brake
        in2.value(1)


# ----- receiving loop ----------------------------------------------------------

def open_link():
    """
    Open the link to the Pi. Returns a function that gives whatever text has
    arrived so far without waiting ("" if nothing).
    """
    global _uart
    if LINK == "uart":
        from machine import UART, Pin
        _uart = UART(UART_ID, baudrate=UART_BAUD, tx=Pin(UART_TX_PIN),
                     rx=Pin(UART_RX_PIN), rxbuf=UART_RXBUF)

        def read_available():
            waiting = _uart.any()
            if not waiting:
                return ""
            data = _uart.read(waiting) or b""
            try:
                return data.decode()
            except UnicodeError:  # noise on the wire: drop it, the line fails to parse
                return "\x00"
        return read_available

    poller = select.poll()
    poller.register(sys.stdin, select.POLLIN)

    def read_available():
        text = ""
        while poller.poll(0):
            text += sys.stdin.read(1)
        return text
    return read_available


def _receive():
    """
    Read everything that has arrived without waiting and return the newest
    valid message as a one-item list ([] if none). Updates _last_rx.

    Each message is a complete snapshot and the Pi repeats a pending command in
    every one, so only the newest line matters: the lines that queued up behind
    it while a move blocked are dropped without parsing. The input is split on
    whole lines, not built up a character at a time, which on the Pico would
    copy the line again for every character.
    """
    global _rx_buffer, _last_rx
    while True:
        text = _reader()
        if not text:
            break
        _rx_buffer += text
    end = _rx_buffer.rfind("\n")
    if end < 0:
        if len(_rx_buffer) > 4000:
            _rx_buffer = ""  # a line that long is garbage; start over
        return []
    lines = _rx_buffer[:end].split("\n")
    _rx_buffer = _rx_buffer[end + 1:]
    for line in reversed(lines):  # newest first; older ones only if it's garbled
        if not line:
            continue
        message = parse(line)
        if message is not None:
            _last_rx = ticks_ms()
            return [message]
        send("ERR bad message")
    return []


def main():
    global _reader, _last_rx, _pending
    setup_motors()
    stop()
    _reader = open_link()
    state = None
    _last_rx = ticks_ms()
    last_status = ticks_ms()
    lost = True
    send("Deskbot Pico ready (%s)" % LINK)

    while True:
        # Handle every message that has arrived. A command can block in
        # on_update(); the lines queued meanwhile are read on the next pass
        # of this inner loop, before the link timeout is checked below, so
        # keep this order.
        while True:
            if _pending is not None:  # a command that cut a spin short
                messages, _pending = [_pending], None
            else:
                messages = _receive()
            if not messages:
                break
            for message in messages:
                state = message
                lost = False
                on_update(state)

        now = ticks_ms()
        if not lost and ticks_diff(now, _last_rx) > LINK_TIMEOUT_MS:
            lost = True
            send("WARN no messages from the Pi, stopping")
            on_lost()

        if ticks_diff(now, last_status) > STATUS_EVERY_MS:
            last_status = now
            if state is not None and not lost:
                bot = state["bot"]
                where = "bot ?" if bot is None else "bot %d,%d %ddeg" % (
                    bot["x"], bot["y"], bot["hd"])
                send("OK seq=%d cal=%d %s objs=%d" % (
                    state["seq"], state["cal"], where, len(state["objs"])))


if __name__ == "__main__":
    main()
