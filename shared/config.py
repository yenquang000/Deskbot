"""
Settings shared by every part of Deskbot: the ArUco markers, the desk coordinate
system and the Pi -> Pico serial link.

Coordinate system (all distances in millimetres):

    marker 0 ──────── +x ───────► marker 1
       │                              │
       +y          desk               │
       │                              │
       ▼                              │
    marker 3 ──────────────────── marker 2

  - The origin (0, 0) is the centre of corner marker 0.
  - x runs toward marker 1, y runs toward marker 3.
  - The desk width/height are the distances between marker centres.
  - Headings are in degrees, 0 = facing +x, 90 = facing +y, range 0-359.

Message sent to the Pico, one JSON object per line (all numbers are integers):

  {"seq": 17, "t": 123456, "desk": [1000, 600], "cal": true,
   "bot": {"x": 412, "y": 230, "hd": 87},
   "objs": [{"id": 3, "x": 610, "y": 145, "w": 80, "h": 60, "a": 4210}]}

  seq   message counter          t     milliseconds since the Pi started
  desk  [width, height] in mm    cal   true once all 4 corners have been seen
  bot   bot marker centre and heading, or null if it is not visible
  objs  flagged objects: centre (x, y), box size along x and y (w, h),
        and the area they actually cover in mm^2 (a)
  When cal is false, bot is null and objs is empty: there are no coordinates yet.

Commands (Pi -> Pico). The Pi is the brain; a message may carry one command in
an optional "cmd" field, and the Pico answers "DONE <id>" when it has finished:

  "cmd": {"id": 81723, "do": "move",  "a": 0}     a = action 0-3
  "cmd": {"id": 81724, "do": "clean", "a": 1500}  a = milliseconds
  "cmd": {"id": 81725, "do": "stop",  "a": null}

  move actions: 0 forward 5 cm, 1 backward 5 cm, 2 heading +45 deg, 3 heading -45 deg
  (the step sizes of Measurements in sim/env.py). +45 makes hd increase.
  - Only one command is pending at a time; a new one replaces an unfinished one.
  - The Pi repeats the pending command in every message until it reads the line
    "DONE <id>" from the Pico, then leaves "cmd" out again.
  - The Pico carries out a command once: it ignores an id equal to the last one
    it did. Ids start at a random number each time the Pi program starts.
"""

# ArUco markers (print them with: python vision/tools/make_markers.py)
ARUCO_DICT = "DICT_4X4_50"
CORNER_IDS = (0, 1, 2, 3)  # origin, +x corner, far corner, +y corner
BOT_ID = 4

# Default desk size: distance between corner marker centres, in mm. This is the
# desk the policy in sim/ was trained on (Measurements in sim/env.py: 120.65 x
# 59.69 cm, rounded to whole mm). Keep the two the same, or retrain.
DESK_WIDTH_MM = 1207
DESK_HEIGHT_MM = 597

# Pi -> Pico serial link (USB). The baud rate is ignored by USB serial but kept
# so the same settings work over UART pins.
SERIAL_BAUD = 115200
DEFAULT_SEND_HZ = 10
LINK_TIMEOUT_S = 0.5  # the Pico stops the bot if no message arrives for this long
