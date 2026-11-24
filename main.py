"""Driver Monitoring System - main program.

Run:   python main.py --driver D01      (or just: python main.py  - it will ask)
Keys:  q = quit, c = recalibrate, m = mute/unmute
Test mode (--test-mode): press the key for what the driver is REALLY doing,
       0 Normal, 1 Drowsy, 2 Distracted, 3 Using Phone, 4 Eating, 5 Drinking.
       It stays selected until you press another key, and is saved as
       True_Label in samples.csv so we can measure accuracy later.
"""
import argparse
import csv
import os
import re
import sys
import time
from datetime import datetime

import cv2
import yaml

from monitor import BEHAVIORS, RISKY
from monitor.alerts import AlertManager
from monitor.calibration import Calibrator
from monitor.camera import FrameSource
from monitor.distraction import DistractionTracker
from monitor.drowsiness import DrowsinessTracker
from monitor.face import FaceAnalyzer
from monitor.head_pose import estimate_head_pose
from monitor.logger import SessionLogger, next_session_id
from monitor.objects import GROUP_TO_BEHAVIOR, ObjectTracker
from monitor.overlay import draw_calibration, draw_face, draw_objects, draw_panel
from monitor.state import FrameSignals, StateMachine

HERE = os.path.dirname(os.path.abspath(__file__))
FACE_MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
                  "face_landmarker/float16/latest/face_landmarker.task")


def parse_args():
    p = argparse.ArgumentParser(description="Driver Monitoring System")
    p.add_argument("--driver", help="driver ID, e.g. D01 (asked at startup if left out)")
    p.add_argument("--session", help="session ID (default: next free S01, S02, ...)")
    p.add_argument("--camera", type=int, help="camera index (default from config.yaml)")
    p.add_argument("--video", help="analyse a recorded video file instead of the webcam")
    p.add_argument("--test-mode", action="store_true", help="enable true-label keys 0-5")
    p.add_argument("--lighting", default="", help="e.g. bright, dim, night")
    p.add_argument("--camera-position", default="", help="e.g. front, left-45, right-45")
    p.add_argument("--glasses", default="", help="yes / no")
    p.add_argument("--notes", default="")
    p.add_argument("--no-audio", action="store_true")
    p.add_argument("--no-objects", action="store_true", help="skip YOLO (faster, face only)")
    p.add_argument("--headless", action="store_true", help="no window (for recorded videos)")
    p.add_argument("--config", default=os.path.join(HERE, "config.yaml"))
    return p.parse_args()


def normalize_driver_id(text):
    """'d2', '2', 'D2' and 'D02' all become 'D02', so one person is never split
    into two drivers in the analysis. Other names are kept, in capitals."""
    t = text.strip().upper().replace(" ", "")
    m = re.fullmatch(r"D?(\d+)", t)
    return f"D{int(m.group(1)):02d}" if m else t


def known_drivers(data_dir):
    path = os.path.join(data_dir, "sessions.csv")
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return sorted({r["Driver_ID"] for r in csv.DictReader(f) if r.get("Driver_ID")})


def get_driver_id(args, data_dir):
    """Driver ID from --driver, or asked in the terminal. Never silently defaults,
    because a wrong ID mixes two people's data in the risk score."""
    if args.driver:
        return normalize_driver_id(args.driver)
    if not sys.stdin or not sys.stdin.isatty():
        sys.exit("No driver ID given. Add --driver D01 (or D02, D03 ...).")
    known = known_drivers(data_dir)
    print("Who is driving?" + (f"  Already recorded: {', '.join(known)}" if known else ""))
    while True:
        driver = normalize_driver_id(input("Driver ID (e.g. D01): "))
        if not driver:
            continue
        if driver not in known and known:
            if input(f"{driver} is a new driver. Correct? [y/n]: ").strip().lower() != "y":
                continue
        return driver


def ensure_face_model(path):
    if os.path.exists(path):
        return
    import urllib.request
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print("Downloading the MediaPipe face model (first run only)...")
    urllib.request.urlretrieve(FACE_MODEL_URL, path)


def main():
    args = parse_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if args.video:
        args.video = os.path.abspath(args.video)
    os.chdir(HERE)  # so relative paths in config.yaml work from anywhere
    data_dir = cfg["logging"]["data_dir"]
    args.driver = get_driver_id(args, data_dir)  # ask first, before the slow model loading

    face_model = os.path.join("models", "face_landmarker.task")
    ensure_face_model(face_model)
    face_analyzer = FaceAnalyzer(face_model)

    detector = None
    if not args.no_objects:
        from monitor.objects import ObjectDetector
        print("Loading YOLO...")
        detector = ObjectDetector(cfg["objects"])

    cam = cfg["camera"]
    source = FrameSource(args.camera if args.camera is not None else cam["index"],
                         cam["width"], cam["height"], cam["mirror"], args.video)
    session_id = args.session or next_session_id(data_dir)
    alerts = AlertManager(audio=not args.no_audio, cooldown=cfg["alerts"]["cooldown_seconds"])
    print(f"Session {session_id}, driver {args.driver}. Press q to quit.")

    run = Runner(cfg, args, session_id, face_analyzer, detector, alerts)
    window = "Driver Monitor"
    try:
        while True:
            frame, t = source.read()
            if frame is None:
                break
            run.step(frame, t)
            if args.headless:
                continue
            cv2.imshow(window, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27) or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break
            run.handle_key(key, t)
    finally:
        run.finish()
        source.release()
        face_analyzer.close()
        cv2.destroyAllWindows()


class Runner:
    """Holds the per-session state and processes one frame at a time."""

    def __init__(self, cfg, args, session_id, face_analyzer, detector, alerts):
        self.cfg, self.args = cfg, args
        self.session_id = session_id
        self.face_analyzer, self.detector, self.alerts = face_analyzer, detector, alerts
        self.objects = ObjectTracker(cfg["objects"])
        self.frame_no = 0
        self.true_label = "Normal" if args.test_mode else ""
        self.fps, self._last_wall = 0.0, None
        self.logger = None
        self.last_t = 0.0
        self.start_calibration()

    # ---- calibration -------------------------------------------------------
    def start_calibration(self):
        d = self.cfg["drowsiness"]
        self.calibrator = Calibrator(self.cfg["calibration"]["seconds"], d["ear_ratio"], d["ear_fallback"])
        self.calib = None

    def finish_calibration(self, t):
        self.calib = self.calibrator.result()
        self.drowsy = DrowsinessTracker(self.cfg["drowsiness"], self.calib["ear_threshold"])
        self.distraction = DistractionTracker(self.cfg["distraction"], self.calib["yaw0"], self.calib["pitch0"])
        self.state = StateMachine(self.cfg["state"]["hold_seconds"])
        if self.logger is None:
            self.t0 = t  # session time starts after calibration
            self.logger = SessionLogger(self.cfg["logging"]["data_dir"], self.session_id, self.args.driver)
            self.next_sample = 0.0
        msg = "OK" if self.calib["ok"] else "face not seen clearly, using default thresholds"
        print(f"Calibration {msg}: EAR baseline={self.calib['ear_baseline']}, "
              f"threshold={self.calib['ear_threshold']}, yaw0={self.calib['yaw0']}, pitch0={self.calib['pitch0']}")

    # ---- per frame ---------------------------------------------------------
    def step(self, frame, t):
        self._update_fps()
        h, w = frame.shape[:2]
        face = self.face_analyzer.process(frame, t)
        pose = estimate_head_pose(face.landmarks, w, h) if face else None
        ear = face.ear if face else None

        if self.calib is None:
            draw_face(frame, face, False)
            if self.calibrator.add(t, ear, pose):
                self.finish_calibration(t)
            else:
                draw_calibration(frame, self.calibrator.progress(t))
            return

        st = t - self.t0
        self.last_t = st
        dz = self.drowsy.update(st, ear)
        ds = self.distraction.update(st, pose)

        if self.detector is not None and self.frame_no % self.cfg["objects"]["every_n_frames"] == 0:
            self.objects.update(self.detector.detect(frame), face.bbox if face else None, st)
        self.frame_no += 1
        obj = self.objects.last

        # While a phone/cup/food is at the face, lowered eyes or a hidden face are
        # caused by the object (sipping, phone at the ear), not by sleep or looking away.
        object_at_face = self.objects.object_at_face_recently(st)
        candidates = {}
        if dz.drowsy and not (object_at_face and dz.trigger == "eye_closure"):
            candidates["Drowsy"] = (dz.confidence, dz.trigger)
        if ds.distracted and not (object_at_face and ds.trigger == "no_face"):
            candidates["Distracted"] = (ds.confidence, ds.trigger)
        for group, conf in obj.active.items():
            names = sorted({d.name for d in obj.detections if d.group == group})
            candidates[GROUP_TO_BEHAVIOR[group]] = (conf, "yolo:" + "/".join(names) if names else "yolo")

        rel = self.distraction.relative(pose[0], pose[1]) if pose else (None, None)
        signals = FrameSignals(st, ear=ear, yaw=rel[0], pitch=rel[1], roll=pose[2] if pose else None,
                               head_pose=ds.head_pose, blink_rate=dz.blink_rate, candidates=candidates)
        finished = self.state.update(signals)
        if finished:
            self.logger.log_event(finished)
        behavior, conf = self.state.behavior, self.state.confidence
        if behavior in RISKY:
            self.alerts.trigger(behavior)

        if st >= self.next_sample:
            self.next_sample += self.cfg["logging"]["sample_every"]
            self.logger.log_sample(st, {
                "Behavior": behavior, "Trigger": self.state.trigger,
                "EAR": _r(ear), "Head_Pose": ds.head_pose, "Yaw": _r(rel[0], 1), "Pitch": _r(rel[1], 1),
                "Roll": _r(pose[2], 1) if pose else None, "Confidence": _r(conf),
                "Alert": "ON" if behavior in RISKY else "OFF", "Blink_Rate": _r(dz.blink_rate, 1),
                "PERCLOS": _r(dz.perclos), "Face_Detected": int(face is not None),
                "Phone_Conf": _r(obj.conf.get("phone", 0)), "Drink_Conf": _r(obj.conf.get("drink", 0)),
                "Food_Conf": _r(obj.conf.get("food", 0)), "True_Label": self.true_label,
                "FPS": _r(self.fps, 1),
                "Raw_Phone_Conf": _r(obj.raw.get("phone", 0)), "Raw_Drink_Conf": _r(obj.raw.get("drink", 0)),
                "Raw_Food_Conf": _r(obj.raw.get("food", 0)), "Object_At_Face": int(object_at_face)})
            self.logger.flush()  # write to disk now, so the Shiny live tab sees it within ~1 s

        # ---- drawing ----
        draw_face(frame, face, dz.eyes_closed)
        draw_objects(frame, obj.detections)
        self.alerts.draw_banner(frame, behavior, conf)
        lines = [f"State: {behavior} ({conf:.0%})",
                 f"EAR: {ear:.3f} (closed < {self.drowsy.threshold:.3f})" if ear else "EAR: no face",
                 f"Blinks/min: {dz.blink_rate:.0f}   PERCLOS: {dz.perclos:.0%}",
                 f"Head: {ds.head_pose}  yaw {rel[0]:+.0f}  pitch {rel[1]:+.0f}" if pose else "Head: no face",
                 f"{self.session_id}  {self.fps:.0f} FPS" + ("" if self.alerts.audio else "  (muted)")]
        if self.args.test_mode:
            lines.append(f"TRUE LABEL: {self.true_label}  (keys 0-5)")
        draw_panel(frame, lines, behavior)

    def handle_key(self, key, t):
        if key == ord("c"):
            print("Recalibrating...")
            self._close_event()
            self.start_calibration()
        elif key == ord("m"):
            self.alerts.audio = not self.alerts.audio
        elif self.args.test_mode and ord("0") <= key <= ord("5"):
            self.true_label = BEHAVIORS[key - ord("0")]

    def _close_event(self):
        if self.calib is not None and self.logger is not None:
            ev = self.state.close(self.last_t)
            if ev and ev.duration > 0:
                self.logger.log_event(ev)

    def _update_fps(self):
        now = time.perf_counter()
        if self._last_wall is not None:
            inst = 1.0 / max(1e-6, now - self._last_wall)
            self.fps = inst if self.fps == 0 else 0.9 * self.fps + 0.1 * inst
        self._last_wall = now

    def finish(self):
        if self.logger is None:
            print("Session ended before calibration finished; nothing was saved.")
            return
        self._close_event()
        c = self.calib or {}
        a = self.args
        self.logger.log_session(self.last_t, {
            "Lighting": a.lighting, "Camera_Position": a.camera_position, "Glasses": a.glasses,
            "Test_Mode": int(a.test_mode), "Source": os.path.basename(a.video) if a.video else "webcam",
            "Calibrated": int(bool(c.get("ok"))), "EAR_Baseline": c.get("ear_baseline"),
            "EAR_Threshold": c.get("ear_threshold"), "Yaw_Baseline": c.get("yaw0"),
            "Pitch_Baseline": c.get("pitch0"), "Notes": a.notes})
        self.logger.close()
        print(f"Saved session {self.session_id}: {self.logger.event_no} events "
              f"({self.logger.risky_events} risky) in {os.path.abspath(self.cfg['logging']['data_dir'])}")


def _r(x, nd=3):
    return None if x is None else round(float(x), nd)


if __name__ == "__main__":
    sys.exit(main())
