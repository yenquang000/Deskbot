"""
Tuning for the vision program.

The constants below are the filter pipeline's defaults (desk_model.py and
filters.py use them directly). The Settings dataclass starts from the same
values and holds the copy the dashboard can change while running; it is saved
to settings.json next to this file when the user clicks Save.
"""

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

from shared import config as shared

MEDIAN_KSIZE = 3                   # median filter before the blur; deletes single-pixel speckles
BLUR_KSIZE = 5                     # Gaussian blur kernel size (odd); smooths sensor noise
LAB_WEIGHTS = (0.2, 1.0, 1.0)      # L (brightness) down-weighted so shadows don't count
MIN_L_FOR_CHROMA = 20.0            # below this lightness, colour (a, b) is too noisy to trust
STD_FLOOR = (4.0, 3.0, 3.0)        # minimum spread per channel; stops hypersensitivity
MAX_L_STD = 30.0                   # learning is rejected if brightness spread exceeds this
SCORE_THRESHOLD = 4.0              # pixel is "not desk" when its score is above this
OPEN_KSIZE = 3                     # opening kernel: removes speckle noise
CLOSE_KSIZE = 7                    # closing kernel: fills holes, joins broken outlines
ROI_FRACTION = 0.8                 # default ROI: centered box, 80% of width and height

SETTINGS_FILE = Path(__file__).with_name("settings.json")


@dataclass
class Settings:
    learn_frames: int = 30  # frames of the clean desk to average over
    blur_kernel: int = BLUR_KSIZE  # Gaussian blur size; must be odd
    std_floor: list = field(default_factory=lambda: list(STD_FLOOR))  # min L, a, b spread
    l_weight: float = LAB_WEIGHTS[0]  # how much brightness counts; low = ignore shadows
    diff_threshold: float = SCORE_THRESHOLD  # score above which a pixel is "not desk"
    open_ksize: int = OPEN_KSIZE  # speckle removal size
    close_ksize: int = CLOSE_KSIZE  # gap filling size
    min_object_area: float = 400.0  # smallest object: mm^2 once calibrated, else pixels
    desk_width_mm: float = shared.DESK_WIDTH_MM  # marker 0 to marker 1, centre to centre
    desk_height_mm: float = shared.DESK_HEIGHT_MM  # marker 0 to marker 3
    send_hz: float = shared.DEFAULT_SEND_HZ  # messages per second to the Pico
    pico_port: str = "auto"  # serial port, or "auto" to find the Pico
    jpeg_quality: int = 80  # quality of the dashboard video stream
    max_fps: int = 30  # cap on frames processed per second

    def validated(self, changes):
        """
        Return a new Settings with `changes` (a dict) applied.
        Raises ValueError with a readable message if a value is out of range.
        """
        data = asdict(self)
        known = {f.name for f in fields(self)}
        for key, value in changes.items():
            if key not in known:
                raise ValueError(f"Unknown setting: {key}")
            data[key] = value

        try:
            new = Settings(
                learn_frames=int(data["learn_frames"]),
                blur_kernel=int(data["blur_kernel"]),
                std_floor=[float(v) for v in data["std_floor"]],
                l_weight=float(data["l_weight"]),
                diff_threshold=float(data["diff_threshold"]),
                open_ksize=int(data["open_ksize"]),
                close_ksize=int(data["close_ksize"]),
                min_object_area=float(data["min_object_area"]),
                desk_width_mm=float(data["desk_width_mm"]),
                desk_height_mm=float(data["desk_height_mm"]),
                send_hz=float(data["send_hz"]),
                pico_port=str(data["pico_port"]).strip() or "auto",
                jpeg_quality=int(data["jpeg_quality"]),
                max_fps=int(data["max_fps"]),
            )
        except (TypeError, ValueError):
            raise ValueError("Settings must be numbers") from None

        if not 1 <= new.learn_frames <= 300:
            raise ValueError("Learn frames must be between 1 and 300")
        if new.blur_kernel < 1 or new.blur_kernel > 31 or new.blur_kernel % 2 == 0:
            raise ValueError("Blur kernel must be an odd number from 1 to 31")
        if len(new.std_floor) != 3 or any(v <= 0 for v in new.std_floor):
            raise ValueError("Std floor needs three positive numbers (L, a, b)")
        if not 0 <= new.l_weight <= 1:
            raise ValueError("Brightness weight must be between 0 and 1")
        if not 0.5 <= new.diff_threshold <= 20:
            raise ValueError("Score threshold must be between 0.5 and 20")
        if not (1 <= new.open_ksize <= 31 and 1 <= new.close_ksize <= 31):
            raise ValueError("Open and close sizes must be between 1 and 31")
        if not 0 <= new.min_object_area <= 1_000_000:
            raise ValueError("Min object area must be between 0 and 1,000,000")
        if not (50 <= new.desk_width_mm <= 10_000 and 50 <= new.desk_height_mm <= 10_000):
            raise ValueError("Desk width and height must be between 50 and 10,000 mm")
        if not 0.5 <= new.send_hz <= 50:
            raise ValueError("Send rate must be between 0.5 and 50 per second")
        if not 10 <= new.jpeg_quality <= 100:
            raise ValueError("JPEG quality must be between 10 and 100")
        if not 1 <= new.max_fps <= 60:
            raise ValueError("Max FPS must be between 1 and 60")
        return new

    def save(self, path=SETTINGS_FILE):
        path.write_text(json.dumps(asdict(self), indent=2) + "\n")


def load(path=SETTINGS_FILE):
    """Load saved settings, falling back to the defaults if the file is missing or bad."""
    try:
        return Settings().validated(json.loads(path.read_text()))
    except (OSError, ValueError):
        return Settings()
