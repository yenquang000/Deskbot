# Deskbot

A small robot that keeps a desk clean. A camera watches the desk, the vision program
finds objects on it and works out where everything is in millimetres, the policy
trained in `sim/` picks the next move from those coordinates, and a Raspberry Pi Pico
carries the move out.

| Folder | What it holds |
|---|---|
| `vision/` | Camera, computer vision, browser dashboard and the Pi → Pico link (see below) |
| `hardware/` | `messaging.py` (Pi side of the serial link, GPIO or USB) and `pico_controller.py` (the Pico's program) |
| `shared/` | `config.py`: marker IDs, coordinate system and message format used by everything |
| `sim/` | Simulation, training, and the control loop that drives the real bot (`pi_runner.py`, `vision_hardware.py`) |

---

## Vision (`vision/`)

### Folder layout

```
vision/
├── main.py              start here: reads command-line options, opens a camera
├── config.py            filter tuning constants and the dashboard's settings
├── requirements.txt
├── camera/
│   ├── cameras.py        OpenCVCamera (webcams, USB) and PiCamera (Pi camera module)
│   └── camera_select.py  OpenCV backends per OS, finding cameras, choosing one
├── cv/
│   ├── desk_model.py     preprocessing and learning the clean desk's color
│   ├── filters.py        frame -> mask: score, threshold, open, close, ROI
│   ├── markers.py        ArUco corner markers -> millimetre grid; bot position
│   └── vision.py         display filters and grouping the mask into objects
├── dashboard/
│   ├── server.py         the Flask web server and its API
│   ├── worker.py         background thread: camera -> detection -> drawing -> Pico
│   ├── drive.py          background thread for --drive: runs the sim/ policy on the bot
│   ├── viewer.py         the plain OpenCV window (--window)
│   ├── templates/        dashboard.html
│   └── static/           dashboard.css, dashboard.js
└── tools/
    ├── make_markers.py   prints the ArUco markers
    ├── test_filters.py   checks for the filter pipeline
    └── test_coordinates.py  checks for markers, coordinates, the Pico link and commands
```

Files created while running (all ignored by git): `vision/settings.json` (saved
settings), `vision/captures/` (snapshots) and `vision/markers/` (printed markers).

### Install

```
cd vision
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt     # opencv-python, numpy, flask, pyserial
```

On the **Raspberry Pi 5**, `picamera2` comes with Raspberry Pi OS (if it's missing:
`sudo apt install -y python3-picamera2`). Create the virtual environment with
`python3 -m venv --system-site-packages .venv` so it can see `picamera2`.

### Run

Always run from inside the `vision` folder:

```
python main.py                    # dashboard; opens your browser at http://127.0.0.1:8000
python main.py --list             # list the cameras that work, then exit
python main.py --camera 1         # use a specific OpenCV camera index
python main.py --picamera         # force the Raspberry Pi camera module
python main.py --width 1280 --height 720
python main.py --host 0.0.0.0     # on the Pi: open http://<pi-ip>:8000 from a laptop
python main.py --port 8080        # use another port
python main.py --no-browser       # don't open a browser tab
python main.py --window           # plain OpenCV window instead of the dashboard
```

With no options it uses the Pi camera module if there is one, otherwise the first
working webcam.

### The dashboard

The live camera is on the left, with object boxes, the bot and the desk grid drawn on
it. Under it are four buttons, which also work as keys (click the page first):

| Button | Key | Does |
|---|---|---|
| Learn | L | Drag a box over clean desk to learn its color (Enter = default area, Esc = cancel) |
| Pause / Resume | P | Freezes the image |
| Snapshot | S | Saves the current frame to `vision/captures/` |
| Quit | Q or Esc | Releases the camera and stops the program |

The right-hand column shows:

- **Filters**: checkboxes that change **only what you see**, never what is detected.
  They can be combined:
  - Blur, L, a, b: views of the camera image (L = lightness, a = green–red,
    b = blue–yellow)
  - Score heatmap, Mask, Highlight differences: the detector's actual results
  - Grid & labels, FPS counter, Learned area: overlays
- **Bot**: its position and heading.
- **Objects**: ID, centre, size and covered area of each flagged object, in mm once
  the desk is calibrated (in pixels before that).
- **Settings** (folded): every tuning value, applied live with **Apply** and kept
  across restarts with **Save**. **Clear desk** forgets the learned desk;
  **Recalibrate corners** looks for the 4 corner markers again.
- **Pico** (folded): connection, messages sent, last message and last reply.
- **Log** (folded): events and errors, including why a learn attempt was rejected.

The top line shows the frame rate, whether the desk is calibrated and whether the Pico
is connected. A warning appears there if the camera sends black frames (for example
when the lens is covered) or stops sending frames.

`--window` opens the plain OpenCV window instead: **l** learns the desk (drag a box,
then Enter), **q** or **Esc** quits. It has no filters, coordinates or Pico link.

### How detection works

The camera may move, so the program doesn't compare each frame with a stored photo.
It learns the clean desk's **color statistics** once and scores every pixel by how far
its color is from the desk:

```
frame -> median filter -> blur -> Lab color -> score vs desk -> threshold
      -> open (remove speckles) -> close (fill gaps) -> desk area -> mask -> objects
```

- **Median filter** removes single-pixel noise, but keeps crumbs a few pixels wide.
- **Lab color** separates brightness (L) from color (a, b). Brightness is weighted low
  (0.2), and a/b are corrected for darkening, so **shadows are not flagged**.
- **Score**: how far a pixel is from the desk color, measured in the desk's own spread.
  Above the threshold (default 4) it counts as "not desk".
- **Desk area**: once the 4 corner markers are found, only the area between them is
  searched, with the markers themselves ignored. Before that, only the learned box.
- **Objects**: each white blob in the mask becomes an object with a box and an ID
  that stays the same between frames. The blob around the bot's marker is the bot and
  isn't listed as an object.

Measured sizes come out about 5 mm larger than the real object (an 80 × 60 mm box
reads about 86 × 65 mm), because the blur and gap filling grow each outline slightly.

#### Learning the desk

**Learn** collects 30 frames (the *Learn frames* setting) and takes every pixel inside
your box. The median color is the desk's color, and the spread of the middle 80% of
pixels is how much the desk normally varies.

Learning is **rejected if the brightness spread is over 30**. That means the box held
more than one surface: the desk edge, the floor, a marker, an object or a hard shadow.
The reason is shown in the Log. Accepting it would make "desk" cover too many colors,
so real objects would be missed. The size of the box doesn't matter, only what's in
it: drag over clear desk surface only, and stay inside the markers.

#### Settings

| Setting | Default | Effect |
|---|---|---|
| Learn frames | 30 | Frames averaged when learning |
| Blur size | 5 | Smooths sensor noise (odd number) |
| Brightness weight | 0.2 | How much brightness counts; lower ignores shadows more |
| Score threshold | 4.0 | Higher = only strongly different pixels are flagged |
| Speckle removal | 3 | Removes flagged spots smaller than this |
| Gap filling | 7 | Joins nearby pieces of one object (a pen and its cap) |
| Min object area | 400 | Smallest object: mm² once calibrated, pixels before |
| Std floor L, a, b | 4, 3, 3 | Stops a very even desk from making detection oversensitive |
| Desk width / height | 1207 / 597 mm | Distance between corner marker centres; the default is the desk the policy was trained on |
| Pico port / sends per second | auto / 10 | Serial port and message rate |
| JPEG quality / Max FPS | 80 / 30 | Dashboard stream quality and processing rate |

The defaults are the constants at the top of `vision/config.py`.

### Setting up the desk

1. Run `python tools/make_markers.py`. It writes the markers to `vision/markers/`.
   Print them all at the same size, keeping the white border around each one.
2. Stick markers 0–3 at the desk corners: 0 = origin, 1 along +x, 2 the far corner,
   3 along +y. Stick marker 4 flat on top of the bot. Marker 5 is the target for
   `--goto` (see below): lay it flat anywhere on the desk.
3. Make sure all four corner markers are fully in the camera's view, white border included.
4. Measure the distance between marker centres (0 → 1 is the width, 0 → 3 the height),
   enter it under **Settings → Desk** and click **Save**.
5. With the desk clear, click **Learn** and drag over the clean desk.

If a corner marker is briefly covered (for example by the bot), the last seen position
is kept. After moving the camera or a marker, click **Recalibrate corners**.

### Coordinates

All positions are in millimetres:

```
marker 0 ──────── +x ───────► marker 1
   │                              │
   +y          desk               │
   ▼                              │
marker 3 ──────────────────── marker 2
```

The origin is the centre of marker 0. Headings are in degrees: 0 = facing +x,
90 = facing +y.

### Messages to the Pico

The Pi sends one JSON line over USB serial, 10 times a second by default:

```json
{"seq":17,"t":123456,"desk":[1000,600],"cal":true,
 "bot":{"x":412,"y":230,"hd":87},
 "objs":[{"id":3,"x":610,"y":145,"w":80,"h":60,"a":4210}]}
```

| Field | Meaning |
|---|---|
| `seq` | Message counter |
| `t` | Milliseconds since the Pi program started |
| `desk` | Desk width and height in mm |
| `cal` | `true` once all 4 corners have been seen; until then `bot` is `null` and `objs` is empty |
| `bot` | Bot centre (`x`, `y`) and heading (`hd`), or `null` if its marker isn't visible |
| `objs` | Each object's ID, centre (`x`, `y`), box size along x and y (`w`, `h`) and covered area (`a`, mm²) |
| `cmd` | Only while a command is pending: `{"id", "do", "a"}`, see *Commands* below |

All numbers are whole numbers. The Pico is found automatically (`/dev/ttyACM*` on
Linux), and the link reconnects by itself if it's unplugged.

#### Commands

When the bot is being driven (`--drive`), a message can also carry one command:

```json
"cmd": {"id": 81723, "do": "move",  "a": 0}
"cmd": {"id": 81724, "do": "clean", "a": 1500}
"cmd": {"id": 81725, "do": "stop",  "a": null}
```

| `do` | `a` | What the Pico does |
|---|---|---|
| `move` | action 0–3 | 0 forward 5 cm, 1 backward 5 cm, 2 heading +45°, 3 heading −45° |
| `clean` | milliseconds | Runs the cleaner for that long |
| `stop` | `null` | Stops the motors |
| `spin` | milliseconds | Test only: both wheels forward for that long (negative = backward, at most 30 s). Stops early if the Pi's messages stop or another command arrives |

The Pi repeats the command in every message until the Pico prints `DONE <id>`, then
leaves `cmd` out again. The Pico carries out each id once. Only one command is pending
at a time, and ids start at a random number each time the program starts. Messages
without a command look exactly as before.

### The Pico

`hardware/pico_controller.py` is the Pico's MicroPython program. Copy it to the Pico
as `main.py` so it runs on power-up:

```
mpremote cp hardware/pico_controller.py :main.py
```

(or open it in Thonny and save it to the Pico as `main.py`). The Pico makes no
decisions: it carries out the commands from the Pi and prints `DONE <id>`. The Pico
calls `stop()` by itself if no message arrives for 0.5 s, and reports back every 2 s;
the replies appear in the dashboard's Pico section.

**Link.** By default (`LINK = "uart"`) the Pico talks to the Pi over GPIO serial:
Pico GP0 → Pi GPIO15 (pin 10), Pico GP1 ← Pi GPIO14 (pin 8), and a shared GND. The Pi
needs `dtparam=uart0=on` in `/boot/firmware/config.txt`, and no serial console on that
port: in `sudo raspi-config` → Interface Options → Serial Port, answer **No** to the login
shell and **Yes** to the serial port hardware (or remove `console=serial0,115200` from
`/boot/firmware/cmdline.txt`), then reboot. Otherwise the console shares the port with
the link and the Pico's replies come back garbled. `messaging.py` then finds
`/dev/ttyAMA0` by itself when no USB Pico is plugged in (this is the Pi 5's GPIO UART; on
a Pi 3 or 4 `/dev/ttyAMA0` is the Bluetooth UART, so set the port explicitly there).
Set `LINK = "usb"` to use the USB cable instead.

**Motors.** A TB6612FNG drives the two N20 motors (wiring from the Drive Test sheet):
GP16 PWMA, GP17/GP18 AIN1/AIN2 (left), GP19/GP20 BIN1/BIN2 (right), GP21 PWMB,
GP22 STBY. Speed ramps up and down (0 to full in about 0.25 s) to keep the start-up
current under the battery's cutoff. The encoders aren't connected yet, so moves are
**timed**: `CM_PER_S` and `DEG_PER_S` at the top of the file are estimates to
calibrate with `python main.py --drive-check` (the file explains the arithmetic).
If a wheel turns the wrong way, set `LEFT_REVERSED` or `RIGHT_REVERSED`. A positive
`turn_deg` makes the camera's heading increase (left wheel forward, right backward).
`PIN_CLEANER` is `None` until the vacuum MOSFET is wired; `clean` then just waits.

**Wheel test** (wheels off the ground; stop the dashboard first, since only one
program can use the link):

```
cd vision
python tools/wheel_test.py              # link check only, nothing moves
python tools/wheel_test.py 0 1 2 3      # each move once: forward, back, turn +, turn -
python tools/wheel_test.py spin 10      # both wheels forward for 10 s (spin -10: backward)
```

### Driving the bot

The trained policy from `sim/` runs on the Pi, inside the vision program:

```
camera → vision (bot pose + objects, mm) → sim/vision_hardware.py (cm, rad)
       → sim/obs.py observation → policy → edge guard → command → Pico moves
       → camera reads the new pose → repeat
```

```
cd vision
python main.py --drive-check                        # test each move against the camera
python main.py --drive ../sim/deskbot_policy.npz    # run the policy
python main.py --goto                               # drive to the target marker (ID 5)
```

| Flag | What it does |
|---|---|
| `--drive PATH_TO_NPZ` | Runs the exported policy on the bot |
| `--drive-check` | Runs each move once (forward, backward, turn +, turn −) and logs the camera-measured distance and heading change against 5 cm and ±45° |
| `--goto` | Drives to the target marker (ID 5) and follows it when it is moved. No policy needed |
| `--clean-on-arrival` | Also sends a `clean` command when a move ends on top of debris (off by default; the cleaner is assumed to run while the bot drives) |
| `--no-guard` | Turns the edge guard off (it is on by default: it overrides any move that would leave the desk) |

Without these flags the program behaves exactly as before and never commands the bot.
The policy needs only NumPy on the Pi. Start, end reason (goal reached, max steps, bot
lost, stopped), steps, coverage, edge guard interventions and errors appear in the
dashboard log. The pose always comes from the camera: if the bot marker or a corner
is lost the bot stops and waits, and gives up after 10 s. The bot marker must sit on
the centre of the robot.

**Go to a marker** (`--goto`). The simplest way to see the bot move under camera
control. Lay marker 5 flat on the desk and the bot drives to it, one move at a time
(the same 5 cm steps and 45° turns as the policy). Each move it either turns toward
the marker, when the marker is more than 30° off its heading, or steps forward. It
stops when its centre is 22 cm from the marker, so it doesn't drive over and hide it
(if the marker does get covered right next to the bot, that counts as arrived).
Move the marker more than 32 cm away and the bot follows. It never steps forward off the
desk: a target too close to an edge makes it wait. If the bot or target marker can't be
seen it waits, and logs once. The target is drawn in magenta on the video and isn't
counted as an object. The steering is in `sim/goto_marker.py`; stop with Ctrl+C.

Objects are split into pens and debris by shape: a pen has a long side of at least
80 mm and is either 3 times longer than wide or fills less than 35% of its box (a
diagonal pen). Pens are pushed to the corner at marker 2. The constants are at the top
of `sim/vision_hardware.py`.

Bring-up order:

1. `--drive-check`. Fix anything it flags first: a step more than 20% off, a turn more
   than 10° off, or a turn in the wrong direction (action 2 must increase the heading).
2. `--goto` with marker 5 on the desk: the bot should drive straight to it.
3. `--drive` with the guard on, on a clear desk.
4. Add debris.
5. Add pens.

The policy was trained on a 120.65 × 59.69 cm desk (±5%), a 10 cm robot radius, 5 cm
steps and 45° turns (`Measurements` in `sim/env.py`). The default marker spacing
(1207 × 597 mm) is that desk, so place the corner markers that far apart. If the marker
spacing set in the dashboard differs by more than 5%, the log shows a warning: match the marker spacing
to the training desk, or set `Measurements` to the real desk and retrain.

### Tests

```
cd vision
python tools/test_filters.py                     # filter pipeline (15 checks)
python tools/test_filters.py --save-debug out    # also writes a picture of every stage
python tools/test_coordinates.py                 # markers, mm coordinates, Pico link, commands, --goto (34 checks)
python tools/test_closed_loop.py                 # whole chain on a simulated bot (16 checks, about 1 min)
cd ../sim
python test_vision_bridge.py                     # vision -> policy / --goto -> commands against the simulator
```

They use generated images, so no camera or Pico is needed. `test_closed_loop.py` runs
the real Pico program (fake `machine.Pin`/`PWM`) on a simulated bot whose wheels follow
the TB6612 pins, renders what the camera would see, and drives it with `--drive-check`,
`--goto` and `--drive` through the real vision code: it checks the pin wiring, the wheel
and turn directions, `LEFT/RIGHT_REVERSED` and the Pico's line reader. They exit with 0 when every
check passes.

### Troubleshooting

| Problem | Fix |
|---|---|
| "No working camera found" | Close other apps using the camera, including another running copy of `main.py` |
| Black image, "Is the lens covered?" warning | Open the webcam's privacy shutter, or press the laptop's camera key |
| "Could not start the dashboard on port 8000" | Another copy is running; stop it or use `--port` |
| Learning rejected | Drag over clear desk surface only (see *Learning the desk*) |
| Corner count stuck below 4/4 | Keep every marker fully in view with its white border, flat and not glossy |
| "Pico not connected" | Plug the Pico in by USB and check it runs `main.py`; set the port in Settings if it isn't found |
| "no DONE from the Pico" in the log | The Pico is unplugged, isn't running `pico_controller.py`, or a move takes longer than 5 s |
| Drive warns about the desk size | Set the marker spacing to the training desk, or change `Measurements` and retrain (see *Driving the bot*) |
| Qt warnings in `--window` mode | Harmless; the window still works |

### Changes to `vision/`

- **Folders.** Code is split into `camera/`, `cv/`, `dashboard/` and `tools/`.
  `camera_backends.py` was merged into `camera/camera_select.py`. The old
  `camera.py` and `inference.py` were replaced.
- **Browser dashboard** is now the default (`--window` keeps the old OpenCV window).
  Every key command is also a button.
- **Filter pipeline** (`cv/desk_model.py`, `cv/filters.py`): weighted Lab color
  score, threshold, opening, closing and ROI. A median filter and a shadow color
  correction were added so shadows and noise are ignored while small crumbs are
  still found. Learning rejects a box that isn't a single surface.
- **Objects** are boxed, given IDs that stay the same between frames, and measured in mm.
- **ArUco markers** (`cv/markers.py`) on the desk corners give a millimetre coordinate
  system; a marker on the bot gives its position and heading.
- **Pico link** (`hardware/messaging.py`, `hardware/pico_controller.py`) sends the
  coordinates over USB 10 times a second.
- **Driving** (`--drive`, `--drive-check`): the policy from `sim/` runs in a thread of
  the vision program and commands the Pico one move at a time.
- **Settings** can be changed live in the dashboard and saved to `vision/settings.json`.
- **Tests** in `vision/tools/` check the filters, coordinates and Pico messages without
  a camera.
