"""Face landmarks (MediaPipe Face Landmarker) and the Eye Aspect Ratio (EAR).

MediaPipe finds 478 points on the face. We use 6 points around each eye:

        p2   p3
    p1          p4        EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)
        p6   p5

An open eye is tall, so EAR is about 0.25-0.35. A closed eye is flat, so EAR
drops towards 0.1. EAR is the same whether the face is near or far from the
camera, because it is a ratio.
"""
from dataclasses import dataclass

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

# Landmark indices, in the order p1..p6 above.
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
MOUTH = [13, 14, 61, 291]


def eye_aspect_ratio(eye_pts):
    """EAR from a (6, 2) array of eye points in the order p1..p6."""
    p1, p2, p3, p4, p5, p6 = np.asarray(eye_pts, dtype=float)
    vertical = np.linalg.norm(p2 - p6) + np.linalg.norm(p3 - p5)
    horizontal = np.linalg.norm(p1 - p4)
    if horizontal < 1e-6:
        return 0.0
    return float(vertical / (2.0 * horizontal))


@dataclass
class FaceResult:
    landmarks: np.ndarray  # (478, 2) pixel coordinates
    ear: float             # average of both eyes
    ear_left: float
    ear_right: float
    bbox: tuple            # (x1, y1, x2, y2) around the face, in pixels

    @property
    def mouth_center(self):
        return self.landmarks[MOUTH].mean(axis=0)


class FaceAnalyzer:
    """Runs the MediaPipe Face Landmarker on video frames."""

    def __init__(self, model_path):
        options = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.landmarker = vision.FaceLandmarker.create_from_options(options)
        self._last_ms = -1

    def process(self, frame_bgr, t):
        """Returns a FaceResult, or None if no face is visible."""
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ms = max(int(t * 1000), self._last_ms + 1)  # timestamps must increase
        self._last_ms = ms
        result = self.landmarker.detect_for_video(image, ms)
        if not result.face_landmarks:
            return None
        pts = np.array([(lm.x * w, lm.y * h) for lm in result.face_landmarks[0]])
        ear_l = eye_aspect_ratio(pts[LEFT_EYE])
        ear_r = eye_aspect_ratio(pts[RIGHT_EYE])
        x1, y1 = pts.min(axis=0)
        x2, y2 = pts.max(axis=0)
        return FaceResult(pts, (ear_l + ear_r) / 2.0, ear_l, ear_r,
                          (float(x1), float(y1), float(x2), float(y2)))

    def close(self):
        self.landmarker.close()
