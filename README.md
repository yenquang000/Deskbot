# Deskbot

A small robot that keeps a desk clean. A camera watches the desk, the vision program
finds objects on it and works out where everything is in millimetres, and a Raspberry
Pi Pico drives the bot using those coordinates.

| Folder | What it holds |
|---|---|
| `vision/` | Camera, computer vision, browser dashboard and the Pi → Pico link (see below) |
| `hardware/` | `messaging.py` (Pi side of the USB link) and `pico_controller.py` (the Pico's program) |
| `shared/` | `config.py`: marker IDs, coordinate system and message format used by everything |
| `sim/` | Simulation and training |

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
│   ├── viewer.py         the plain OpenCV window (--window)
│   ├── templates/        dashboard.html
│   └── static/           dashboard.css, dashboard.js
└── tools/
    ├── make_markers.py   prints the ArUco markers
    ├── test_filters.py   checks for the filter pipeline
    └── test_coordinates.py  checks for markers, coordinates and the Pico link
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
| Desk width / height | 1000 / 600 mm | Distance between corner marker centres |
| Pico port / sends per second | auto / 10 | Serial port and message rate |
| JPEG quality / Max FPS | 80 / 30 | Dashboard stream quality and processing rate |

The defaults are the constants at the top of `vision/config.py`.

### Setting up the desk

1. Run `python tools/make_markers.py`. It writes the markers to `vision/markers/`.
   Print them all at the same size, keeping the white border around each one.
2. Stick markers 0–3 at the desk corners: 0 = origin, 1 along +x, 2 the far corner,
   3 along +y. Stick marker 4 flat on top of the bot.
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

All numbers are whole numbers. The Pico is found automatically (`/dev/ttyACM*` on
Linux), and the link reconnects by itself if it's unplugged.

### The Pico

`hardware/pico_controller.py` is the Pico's MicroPython program. Copy it to the Pico
as `main.py` so it runs on power-up:

```
mpremote cp hardware/pico_controller.py :main.py
```

(or open it in Thonny and save it to the Pico as `main.py`). All movement
calculations go in `on_update()`, and `stop()` should stop the motors. The Pico calls
`stop()` by itself if no message arrives for 0.5 s, and reports back every 2 s; the
replies appear in the dashboard's Pico section.

### Tests

```
cd vision
python tools/test_filters.py                     # filter pipeline (15 checks)
python tools/test_filters.py --save-debug out    # also writes a picture of every stage
python tools/test_coordinates.py                 # markers, mm coordinates, Pico link (17 checks)
```

Both use generated images, so no camera or Pico is needed. They exit with 0 when every
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
- **Settings** can be changed live in the dashboard and saved to `vision/settings.json`.
- **Tests** in `vision/tools/` check the filters, coordinates and Pico messages without
  a camera.
