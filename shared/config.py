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
"""

# ArUco markers (print them with: python vision/tools/make_markers.py)
ARUCO_DICT = "DICT_4X4_50"
CORNER_IDS = (0, 1, 2, 3)  # origin, +x corner, far corner, +y corner
BOT_ID = 4

# Default desk size: distance between corner marker centres, in mm.
DESK_WIDTH_MM = 1000
DESK_HEIGHT_MM = 600

# Pi -> Pico serial link (USB). The baud rate is ignored by USB serial but kept
# so the same settings work over UART pins.
SERIAL_BAUD = 115200
DEFAULT_SEND_HZ = 10
LINK_TIMEOUT_S = 0.5  # the Pico stops the bot if no message arrives for this long
