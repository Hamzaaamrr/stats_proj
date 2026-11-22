"""Unit tests for the detection logic (no webcam needed).  Run:  python -m pytest -q"""
import math
import os
import sys

import cv2
import numpy as np
import pytest
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from monitor.calibration import Calibrator
from monitor.distraction import DistractionTracker
from monitor.drowsiness import DrowsinessTracker
from monitor.face import eye_aspect_ratio
from monitor.head_pose import MODEL_POINTS, POSE_IDS, camera_matrix, estimate_head_pose
from monitor.logger import SessionLogger, next_session_id
from monitor.objects import Detection, ObjectTracker
from monitor.state import FrameSignals, StateMachine

CFG = yaml.safe_load(open(os.path.join(os.path.dirname(__file__), "..", "config.yaml"), encoding="utf-8"))
FPS = 30


def run_ear(tracker, ears, t0=0.0):
    status = None
    for i, e in enumerate(ears):
        status = tracker.update(t0 + i / FPS, e)
    return status


# ---- EAR -------------------------------------------------------------------
def test_ear_open_vs_closed():
    open_eye = [(0, 0), (10, -6), (20, -6), (30, 0), (20, 6), (10, 6)]
    closed_eye = [(0, 0), (10, -1), (20, -1), (30, 0), (20, 1), (10, 1)]
    assert eye_aspect_ratio(open_eye) == pytest.approx(0.4)
    assert eye_aspect_ratio(closed_eye) == pytest.approx(2 / 30)


def test_ear_is_scale_invariant():
    eye = np.array([(0, 0), (10, -6), (20, -6), (30, 0), (20, 6), (10, 6)], float)
    assert eye_aspect_ratio(eye) == pytest.approx(eye_aspect_ratio(eye * 3.7))


# ---- Drowsiness ------------------------------------------------------------
def test_short_closures_are_blinks_not_drowsy():
    tr = DrowsinessTracker(CFG["drowsiness"], 0.21)
    pattern = [0.3] * 20 + [0.1] * 5  # 0.17 s closure, every 0.83 s
    s = run_ear(tr, pattern * 12 + [0.3])
    assert not s.drowsy or s.trigger == "blink_rate"
    assert len(tr.blinks) == 12


def test_long_closure_is_drowsy():
    tr = DrowsinessTracker(CFG["drowsiness"], 0.21)
    s = run_ear(tr, [0.3] * 30 + [0.12] * int(1.6 * FPS))
    assert s.drowsy and s.trigger == "eye_closure"
    assert 0.6 <= s.confidence <= 1.0


def test_closure_just_under_limit_is_not_drowsy():
    tr = DrowsinessTracker(CFG["drowsiness"], 0.21)
    s = run_ear(tr, [0.3] * 30 + [0.12] * int(1.3 * FPS))
    assert not s.drowsy


def test_high_blink_rate_is_drowsy_after_window():
    tr = DrowsinessTracker(CFG["drowsiness"], 0.21)
    pattern = [0.3] * 40 + [0.1] * 4  # ~41 blinks/min
    s = run_ear(tr, pattern * 25)     # ~37 s
    assert s.drowsy and s.trigger == "blink_rate"


def test_perclos():
    tr = DrowsinessTracker(CFG["drowsiness"], 0.21)
    pattern = [0.3] * 18 + [0.1] * 12  # closed 40% of the time, closures 0.4 s
    s = run_ear(tr, pattern * 70)     # 70 s
    assert s.perclos == pytest.approx(0.4, abs=0.02)
    assert s.drowsy


# ---- Head pose -------------------------------------------------------------
def _project(yaw, pitch, w=640, h=480):
    base = cv2.Rodrigues(np.array([math.pi, 0.0, 0.0]))[0]
    a, b = math.radians(yaw), math.radians(pitch)
    Ry = np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])
    Rx = np.array([[1, 0, 0], [0, math.cos(b), -math.sin(b)], [0, math.sin(b), math.cos(b)]])
    rvec = cv2.Rodrigues(base @ Ry @ Rx)[0]
    pts, _ = cv2.projectPoints(MODEL_POINTS, rvec, np.array([0, 0, 2000.0]), camera_matrix(w, h), None)
    lm = np.zeros((478, 2))
    lm[POSE_IDS] = pts.reshape(-1, 2)
    return lm


@pytest.mark.parametrize("yaw,pitch", [(0, 0), (35, 0), (-35, 0), (0, 25), (0, -20), (20, 10)])
def test_head_pose_recovers_angles(yaw, pitch):
    y, p, _ = estimate_head_pose(_project(yaw, pitch), 640, 480)
    assert y == pytest.approx(yaw, abs=1.0)
    assert p == pytest.approx(pitch, abs=1.0)


# ---- Distraction -----------------------------------------------------------
def test_distraction_needs_two_seconds_and_uses_baseline():
    tr = DistractionTracker(CFG["distraction"], yaw0=-20, pitch0=0)  # camera off to one side
    # -20 deg is the driver's normal pose here, so not distracted:
    assert tr.update(0.0, (-20, 0, 0)).head_pose == "Forward"
    assert not tr.update(1.0, (20, 0, 0)).distracted   # +40 relative, but only just started
    s = tr.update(3.1, (20, 0, 0))
    assert s.distracted and s.head_pose == "Right" and s.trigger == "head_turn"


def test_no_face_becomes_distracted():
    tr = DistractionTracker(CFG["distraction"])
    assert not tr.update(0.0, None).distracted
    s = tr.update(2.5, None)
    assert s.distracted and s.head_pose == "No Face"


# ---- Calibration -----------------------------------------------------------
def test_calibration_median_and_threshold():
    c = Calibrator(5, 0.75, 0.21)
    for i in range(150):
        c.add(i / 30, 0.30, (10.0, -5.0, 0.0))
    r = c.result()
    assert r["ok"] and r["ear_threshold"] == pytest.approx(0.225) and r["yaw0"] == 10.0


def test_calibration_falls_back_without_face():
    c = Calibrator(5, 0.75, 0.21)
    for i in range(160):
        c.add(i / 30, None, None)
    assert c.done and not c.result()["ok"] and c.result()["ear_threshold"] == 0.21


# ---- Objects ---------------------------------------------------------------
FACE = (200, 100, 400, 350)


def test_phone_needs_most_recent_frames():
    tr = ObjectTracker(CFG["objects"])
    phone = Detection("cell phone", "phone", 0.8, (450, 150, 520, 260))
    for i in range(15):
        st = tr.update([phone] if i % 3 else [], FACE)  # 2/3 of frames
    assert "phone" in st.active
    tr2 = ObjectTracker(CFG["objects"])
    for i in range(15):
        st = tr2.update([phone] if i % 3 == 0 else [], FACE)  # 1/3 of frames
    assert "phone" not in st.active


def test_cup_far_from_face_is_ignored():
    tr = ObjectTracker(CFG["objects"])
    far_cup = Detection("cup", "drink", 0.9, (600, 440, 640, 480))
    near_cup = Detection("cup", "drink", 0.9, (300, 250, 360, 330))
    for _ in range(15):
        st = tr.update([far_cup], FACE)
    assert "drink" not in st.active
    for _ in range(15):
        st = tr.update([near_cup], FACE)
    assert st.active["drink"] == pytest.approx(0.9)


# ---- State machine ---------------------------------------------------------
def test_priority_and_events():
    sm = StateMachine(hold_seconds=0.5)
    ev = []
    def step(t, cands):
        e = sm.update(FrameSignals(t, ear=0.3, yaw=0, pitch=0, roll=0, head_pose="Forward", candidates=cands))
        if e:
            ev.append(e)
    for i in range(30):
        step(i / 10, {})
    for i in range(30, 60):  # phone and eating at once -> phone wins
        step(i / 10, {"Using Phone": (0.8, "yolo"), "Eating": (0.5, "yolo")})
    assert sm.behavior == "Using Phone"
    step(6.1, {"Drowsy": (0.9, "eye_closure"), "Using Phone": (0.8, "yolo")})
    assert sm.behavior == "Drowsy"
    for i in range(62, 80):
        step(i / 10, {})
    assert sm.behavior == "Normal"
    assert [e.behavior for e in ev] == ["Normal", "Using Phone", "Drowsy"]
    assert ev[1].duration == pytest.approx(3.1)
    assert ev[1].summary()["Alert"] == "ON"


def test_hold_prevents_flicker():
    sm = StateMachine(hold_seconds=0.7)
    sm.update(FrameSignals(0.0, candidates={"Drinking": (0.7, "yolo")}))
    sm.update(FrameSignals(0.3, candidates={}))
    assert sm.behavior == "Drinking"
    sm.update(FrameSignals(1.0, candidates={}))
    assert sm.behavior == "Normal"


# ---- Logger ----------------------------------------------------------------
def test_logger_writes_csvs_and_session_ids(tmp_path):
    d = str(tmp_path)
    assert next_session_id(d) == "S01"
    sm = StateMachine()
    sm.update(FrameSignals(0.0, ear=0.3, candidates={"Drowsy": (0.9, "eye_closure")}))
    ev = sm.close(2.0)
    lg = SessionLogger(d, "S01", "D01")
    lg.log_event(ev)
    lg.log_sample(0.0, {"Behavior": "Drowsy", "True_Label": "Drowsy"})
    lg.log_session(2.0, {"Lighting": "dim"})
    lg.close()
    assert next_session_id(d) == "S02"
    rows = open(os.path.join(d, "events.csv"), encoding="utf-8").read().splitlines()
    assert rows[0].startswith("Session_ID,Event_ID") and ",Drowsy,eye_closure," in rows[1]
