"""Head pose (yaw, pitch, roll) from 6 face landmarks with OpenCV solvePnP.

We know roughly where the nose, chin, eye corners and mouth corners sit on an
average 3D head. solvePnP finds the rotation of that 3D head that best lines up
with the 2D points the camera sees. From the rotation we get:

    yaw   = turning left / right   (positive = driver's right, with a mirrored image)
    pitch = nodding down / up      (positive = looking down)
    roll  = tilting ear to shoulder
"""
import math

import cv2
import numpy as np

# Landmark ids: nose tip, chin, eye outer corners (image left, image right),
# mouth corners (image left, image right).
POSE_IDS = [1, 152, 33, 263, 61, 291]

# Matching points on a generic 3D head (x right, y up, z towards the camera).
MODEL_POINTS = np.array([
    (0.0, 0.0, 0.0),          # nose tip
    (0.0, -330.0, -65.0),     # chin
    (-225.0, 170.0, -135.0),  # eye corner, image left
    (225.0, 170.0, -135.0),   # eye corner, image right
    (-150.0, -150.0, -125.0), # mouth corner, image left
    (150.0, -150.0, -125.0),  # mouth corner, image right
], dtype=np.float64)


def camera_matrix(width, height):
    """Simple pinhole camera: focal length about the image width."""
    f = float(width)
    return np.array([[f, 0, width / 2.0], [0, f, height / 2.0], [0, 0, 1]], dtype=np.float64)


def angles_from_rotation(rvec, image_points):
    """Turn a solvePnP rotation into (yaw, pitch, roll) in degrees."""
    R, _ = cv2.Rodrigues(rvec)
    forward = R @ np.array([0.0, 0.0, 1.0])  # where the nose points, in camera space
    fx, fy, fz = forward
    yaw = math.degrees(math.atan2(fx, -fz))
    pitch = math.degrees(math.atan2(fy, math.hypot(fx, fz)))
    # Roll from the line between the two outer eye corners.
    (lx, ly), (rx, ry) = image_points[2], image_points[3]
    roll = math.degrees(math.atan2(ry - ly, rx - lx))
    return yaw, pitch, roll


def estimate_head_pose(landmarks, width, height):
    """Returns (yaw, pitch, roll) in degrees, or None if solvePnP fails."""
    image_points = np.asarray(landmarks, dtype=np.float64)[POSE_IDS]
    ok, rvec, _tvec = cv2.solvePnP(
        MODEL_POINTS, image_points, camera_matrix(width, height),
        np.zeros((4, 1)), flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None
    return angles_from_rotation(rvec, image_points)


def pose_label(yaw, pitch, cfg):
    """Names the head direction after removing the calibration baseline."""
    if pitch > cfg["pitch_down_threshold"]:
        return "Down"
    if pitch < -cfg["pitch_up_threshold"]:
        return "Up"
    if yaw > cfg["yaw_threshold"]:
        return "Right"
    if yaw < -cfg["yaw_threshold"]:
        return "Left"
    return "Forward"
