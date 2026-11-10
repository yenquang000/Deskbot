"""
ArUco markers: the four desk corners define the desk coordinate system (mm), and
the marker on the bot gives its position and heading. See shared/config.py for
the coordinate convention.
"""

import math

import cv2
import numpy as np

from shared import config as shared

EDGE_PAD = 16  # pixels added around the frame before looking for markers


class DeskFrame:
    """The result of one marker update: calibration, bot pose and marker outlines."""

    def __init__(self, homography, size_mm, corners_px, visible, bot, outlines):
        self.homography = homography  # pixels -> mm, or None if not calibrated
        self.inverse = None if homography is None else np.linalg.inv(homography)
        self.size_mm = size_mm  # (width, height)
        self.corners_px = corners_px  # {id: (x, y)} last known centre of each corner
        self.visible = visible  # corner ids seen in this frame
        self.bot = bot  # {"x", "y", "hd", "px": (x, y), "tip_px": (x, y)} or None
        self.outlines = outlines  # {marker id: pixel outline (4x2)} seen in this frame

    @property
    def calibrated(self):
        return self.homography is not None

    def to_desk(self, points_px):
        """Pixel points (N x 2) -> desk mm (N x 2)."""
        pts = np.asarray(points_px, dtype=np.float32).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(pts, self.homography).reshape(-1, 2)

    def to_pixels(self, points_mm):
        """Desk mm points (N x 2) -> pixels (N x 2)."""
        pts = np.asarray(points_mm, dtype=np.float32).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(pts, self.inverse).reshape(-1, 2)

    def desk_polygon_px(self):
        """The desk area in pixels, as an int polygon for drawing and masking."""
        w, h = self.size_mm
        return self.to_pixels([(0, 0), (w, 0), (w, h), (0, h)]).round().astype(np.int32)


class MarkerTracker:
    """Finds the markers each frame and remembers where the corners were last seen."""

    def __init__(self, desk_width_mm=shared.DESK_WIDTH_MM, desk_height_mm=shared.DESK_HEIGHT_MM):
        dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, shared.ARUCO_DICT))
        self.detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())
        self.size_mm = (desk_width_mm, desk_height_mm)
        self.corners_px = {}  # remembered so a briefly covered corner keeps calibration

    def set_desk_size(self, width_mm, height_mm):
        self.size_mm = (width_mm, height_mm)

    def reset(self):
        """Forget the remembered corners (use after moving the camera or the markers)."""
        self.corners_px = {}

    def update(self, frame):
        # The detector needs some quiet space around each marker; a corner marker
        # near the edge of the camera view is often missed without it. Extending
        # the frame's edge pixels outward gives it that space.
        padded = cv2.copyMakeBorder(frame, EDGE_PAD, EDGE_PAD, EDGE_PAD, EDGE_PAD,
                                    cv2.BORDER_REPLICATE)
        corners, ids, _ = self.detector.detectMarkers(padded)
        found = {}
        if ids is not None:
            for quad, marker_id in zip(corners, ids.flatten()):
                found[int(marker_id)] = quad.reshape(4, 2) - EDGE_PAD

        visible = [i for i in shared.CORNER_IDS if i in found]
        for i in visible:
            self.corners_px[i] = tuple(found[i].mean(axis=0))

        homography = None
        if all(i in self.corners_px for i in shared.CORNER_IDS):
            w, h = self.size_mm
            src = np.float32([self.corners_px[i] for i in shared.CORNER_IDS])
            dst = np.float32([(0, 0), (w, 0), (w, h), (0, h)])
            homography = cv2.getPerspectiveTransform(src, dst)

        frame_info = DeskFrame(homography, self.size_mm, dict(self.corners_px), visible,
                               None, found)
        if homography is not None and shared.BOT_ID in found:
            frame_info.bot = self._bot_pose(frame_info, found[shared.BOT_ID])
        return frame_info

    @staticmethod
    def _bot_pose(frame_info, quad_px):
        """Bot centre in mm and heading: the direction the marker's top edge faces."""
        # ArUco corner order: top-left, top-right, bottom-right, bottom-left.
        quad_mm = frame_info.to_desk(quad_px)
        centre = quad_mm.mean(axis=0)
        top_mid = (quad_mm[0] + quad_mm[1]) / 2
        dx, dy = top_mid - centre
        heading = math.degrees(math.atan2(dy, dx)) % 360
        centre_px = quad_px.mean(axis=0)
        tip_px = centre_px + 2.5 * ((quad_px[0] + quad_px[1]) / 2 - centre_px)
        return {
            "x": float(centre[0]),
            "y": float(centre[1]),
            "hd": heading,
            "px": tuple(centre_px),
            "tip_px": tuple(tip_px),
            "quad_px": quad_px,
        }
