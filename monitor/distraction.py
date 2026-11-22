"""Distraction: head turned away from the road, or face not visible, for too long.

Angles are measured relative to the driver's own calibrated "looking at the
road" pose, so a camera placed at an angle still works.
"""
from dataclasses import dataclass

from .head_pose import pose_label


@dataclass
class DistractionStatus:
    head_pose: str      # Forward / Left / Right / Down / Up / No Face
    away_for: float     # seconds the driver has been looking away
    distracted: bool
    confidence: float
    trigger: str        # "head_turn", "no_face" or ""


class DistractionTracker:
    def __init__(self, cfg, yaw0=0.0, pitch0=0.0):
        self.cfg = cfg
        self.yaw0, self.pitch0 = yaw0, pitch0
        self.away_since = None
        self.noface_since = None

    def relative(self, yaw, pitch):
        return yaw - self.yaw0, pitch - self.pitch0

    def update(self, t, pose):
        """pose = (yaw, pitch, roll) in degrees, or None when no face is visible."""
        cfg = self.cfg
        if pose is None:
            self.away_since = None
            if self.noface_since is None:
                self.noface_since = t
            gone = t - self.noface_since
            if gone >= cfg["no_face_seconds"]:
                return DistractionStatus("No Face", gone, True, 0.6, "no_face")
            return DistractionStatus("No Face", gone, False, 0.0, "")

        self.noface_since = None
        yaw, pitch = self.relative(pose[0], pose[1])
        label = pose_label(yaw, pitch, cfg)
        if label == "Forward":
            self.away_since = None
            return DistractionStatus(label, 0.0, False, 0.0, "")
        if self.away_since is None:
            self.away_since = t
        away = t - self.away_since
        if away < cfg["seconds"]:
            return DistractionStatus(label, away, False, 0.0, "")
        # Confidence grows with how far past the threshold the head is turned.
        if label in ("Left", "Right"):
            excess = (abs(yaw) - cfg["yaw_threshold"]) / cfg["yaw_threshold"]
        elif label == "Down":
            excess = (pitch - cfg["pitch_down_threshold"]) / cfg["pitch_down_threshold"]
        else:
            excess = (-pitch - cfg["pitch_up_threshold"]) / cfg["pitch_up_threshold"]
        longer = (away - cfg["seconds"]) / cfg["seconds"]
        conf = 0.6 + 0.25 * min(1.0, max(0.0, excess)) + 0.15 * min(1.0, longer)
        return DistractionStatus(label, away, True, round(conf, 3), "head_turn")
