"""Driver Monitoring System - main program.

Run:   python main.py --driver D01
Keys:  q/Esc = stop, c = recalibrate, m = mute/unmute
Test mode (--test-mode): runs temporarily without saving session, event, or sample
       data. Press 0-5 to set the true label shown by the desktop preview.
"""
import argparse
import os
import queue
import sys
import threading
import time
import traceback
import webbrowser

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
    p.add_argument("--driver", default="D01", help="driver ID, e.g. D01")
    p.add_argument("--session", help="session ID (default: next free S01, S02, ...)")
    p.add_argument("--camera", type=int, help="camera index (default from config.yaml)")
    p.add_argument("--video", help="analyse a recorded video file instead of the webcam")
    p.add_argument("--test-mode", action="store_true",
                   help="run temporarily without saving session data; enable true-label keys 0-5")
    p.add_argument("--lighting", default="", help="e.g. bright, dim, night")
    p.add_argument("--camera-position", default="", help="e.g. front, left-45, right-45")
    p.add_argument("--glasses", default="", help="yes / no")
    p.add_argument("--notes", default="")
    p.add_argument("--no-audio", action="store_true")
    p.add_argument("--no-objects", action="store_true", help="skip YOLO (faster, face only)")
    p.add_argument("--desktop", action="store_true", help="use the desktop dashboard instead of the web app")
    p.add_argument("--port", type=int, default=8765, help="local web dashboard port")
    p.add_argument("--no-browser", action="store_true", help="do not open the web dashboard automatically")
    p.add_argument("--headless", action="store_true", help="run without the desktop GUI")
    p.add_argument("--config", default=os.path.join(HERE, "config.yaml"))
    return p.parse_args()


def ensure_face_model(path):
    if os.path.exists(path):
        return
    import urllib.request
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print("Downloading the MediaPipe face model (first run only)...")
    urllib.request.urlretrieve(FACE_MODEL_URL, path)


def main():
    args = parse_args()
    if not 1 <= args.port <= 65535:
        raise SystemExit("--port must be between 1 and 65535.")
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if args.video:
        args.video = os.path.abspath(args.video)
    os.chdir(HERE)  # so relative paths in config.yaml work from anywhere

    if args.headless:
        run_headless(cfg, args)
        return 0

    if args.desktop:
        from monitor.gui import DriverMonitorApp

        app = DriverMonitorApp(cfg, args, lambda settings: MonitorWorker(cfg, settings))
        app.run()
    else:
        from monitor.web import create_app
        from waitress import serve

        defaults = {
            "driver": args.driver,
            "camera": args.camera if args.camera is not None else cfg["camera"]["index"],
            "lighting": args.lighting,
            "camera_position": args.camera_position,
            "glasses": args.glasses,
            "test_mode": args.test_mode,
            "objects": not args.no_objects,
            "audio": not args.no_audio,
            "video": args.video or "",
        }
        app = create_app(
            lambda settings: MonitorWorker(cfg, settings),
            defaults,
            args.video or "",
            os.path.abspath(cfg["logging"]["data_dir"]),
        )
        url = f"http://127.0.0.1:{args.port}"
        print(f"Local-only dashboard: {url}")
        if not args.no_browser:
            threading.Timer(0.8, webbrowser.open, args=(url,)).start()
        serve(app, host="127.0.0.1", port=args.port, threads=8)
    return 0


def run_headless(cfg, args):
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
    data_dir = cfg["logging"]["data_dir"]
    session_id = args.session or next_session_id(data_dir)
    alerts = AlertManager(audio=not args.no_audio, cooldown=cfg["alerts"]["cooldown_seconds"])
    print(f"Session {session_id}, driver {args.driver}. Press q to quit.")

    run = Runner(cfg, args, session_id, face_analyzer, detector, alerts)
    try:
        while True:
            frame, t = source.read()
            if frame is None:
                break
            run.step(frame, t)
    finally:
        run.finish()
        source.release()
        face_analyzer.close()


class MonitorWorker:
    """Runs camera capture and detection away from the GUI event loop."""

    def __init__(self, cfg, settings):
        self.cfg = cfg
        self.settings = settings
        self.stop_event = threading.Event()
        self.commands = queue.Queue()
        self.lock = threading.Lock()
        self.frame = None
        self.frame_version = 0
        self.metrics = {}
        self.message = "Starting monitor..."
        self.error = ""
        self.finished = False
        self.saved = False
        self.session_id = None
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self.thread.start()

    def stop(self):
        self.stop_event.set()

    def command(self, key):
        self.commands.put(key)

    def snapshot(self):
        with self.lock:
            return {
                "frame": self.frame,
                "frame_version": self.frame_version,
                "metrics": dict(self.metrics),
                "message": self.message,
                "error": self.error,
                "finished": self.finished,
                "saved": self.saved,
                "session_id": self.session_id,
            }

    def _set_message(self, message):
        with self.lock:
            self.message = message

    def _run(self):
        face_analyzer = None
        source = None
        run = None
        failure = None
        try:
            self._set_message("Loading face detection model...")
            face_model = os.path.join(HERE, "models", "face_landmarker.task")
            ensure_face_model(face_model)
            face_analyzer = FaceAnalyzer(face_model)

            detector = None
            if self.settings["objects"]:
                from monitor.objects import ObjectDetector
                self._set_message("Loading object detection model...")
                detector = ObjectDetector(self.cfg["objects"])

            self._set_message("Opening camera...")
            camera = self.cfg["camera"]
            source = FrameSource(
                self.settings["camera"],
                camera["width"],
                camera["height"],
                camera["mirror"],
                self.settings["video"] or None,
            )
            data_dir = self.cfg["logging"]["data_dir"]
            self.session_id = next_session_id(data_dir)
            args = argparse.Namespace(
                driver=self.settings["driver"],
                test_mode=self.settings["test_mode"],
                no_audio=not self.settings["audio"],
                no_objects=not self.settings["objects"],
                video=self.settings["video"] or None,
                dashboard=True,
                lighting=self.settings["lighting"],
                camera_position=self.settings["camera_position"],
                glasses=self.settings["glasses"],
                notes="",
            )
            alerts = AlertManager(
                audio=self.settings["audio"],
                cooldown=self.cfg["alerts"]["cooldown_seconds"],
            )
            run = Runner(self.cfg, args, self.session_id, face_analyzer, detector, alerts)
            self._set_message(f"Monitoring session {self.session_id}")

            while not self.stop_event.is_set():
                while True:
                    try:
                        key = self.commands.get_nowait()
                    except queue.Empty:
                        break
                    run.handle_key(key, run.last_t)

                frame, t = source.read()
                if frame is None:
                    break
                run.step(frame, t)
                with self.lock:
                    self.frame = frame.copy()
                    self.frame_version += 1
                    self.metrics = run.snapshot()
        except Exception as exc:
            failure = exc
            traceback.print_exc()
        finally:
            if run is not None:
                try:
                    run.finish()
                except Exception as exc:
                    if failure is None:
                        failure = exc
                    traceback.print_exc()
            if source is not None:
                try:
                    source.release()
                except Exception as exc:
                    if failure is None:
                        failure = exc
                    traceback.print_exc()
            if face_analyzer is not None:
                try:
                    face_analyzer.close()
                except Exception as exc:
                    if failure is None:
                        failure = exc
                    traceback.print_exc()
            temporary_video = self.settings.get("temporary_video")
            if temporary_video:
                try:
                    os.remove(temporary_video)
                except FileNotFoundError:
                    pass
                except OSError as exc:
                    if failure is None:
                        failure = exc
                    traceback.print_exc()
            with self.lock:
                self.error = str(failure) if failure is not None else ""
                self.saved = run is not None and run.logger is not None and failure is None
                if self.saved:
                    self.message = f"Session {self.session_id} saved."
                elif run is not None and self.settings["test_mode"] and failure is None:
                    self.message = f"Test session {self.session_id} ended; nothing was saved."
                else:
                    self.message = "Monitor stopped."
                self.finished = True


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
        self.metrics = {
            "behavior": "Calibrating",
            "confidence": 0.0,
            "calibration_progress": 0.0,
            "ear": None,
            "blink_rate": 0.0,
            "perclos": 0.0,
            "head_pose": "Calibrating",
            "yaw": None,
            "pitch": None,
            "fps": 0.0,
        }
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
            self.t0 = t
            if not self.args.test_mode:
                self.logger = SessionLogger(
                    self.cfg["logging"]["data_dir"], self.session_id, self.args.driver)
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
            calibrated = self.calibrator.add(t, ear, pose)
            self.metrics.update({
                "behavior": "Calibrating",
                "calibration_progress": self.calibrator.progress(t),
                "ear": ear,
                "face_detected": face is not None,
                "fps": self.fps,
            })
            if calibrated:
                self.finish_calibration(t)
            else:
                draw_calibration(frame, self.calibrator.progress(t))
            return

        st = t - self.t0
        self.last_t = st
        dz = self.drowsy.update(st, ear)
        ds = self.distraction.update(st, pose)

        if self.detector is not None and self.frame_no % self.cfg["objects"]["every_n_frames"] == 0:
            self.objects.update(self.detector.detect(frame), face.bbox if face else None)
        self.frame_no += 1
        obj = self.objects.last

        candidates = {}
        if dz.drowsy:
            candidates["Drowsy"] = (dz.confidence, dz.trigger)
        if ds.distracted:
            candidates["Distracted"] = (ds.confidence, ds.trigger)
        for group, conf in obj.active.items():
            names = sorted({d.name for d in obj.detections if d.group == group})
            candidates[GROUP_TO_BEHAVIOR[group]] = (conf, "yolo:" + "/".join(names) if names else "yolo")

        rel = self.distraction.relative(pose[0], pose[1]) if pose else (None, None)
        signals = FrameSignals(st, ear=ear, yaw=rel[0], pitch=rel[1], roll=pose[2] if pose else None,
                               head_pose=ds.head_pose, blink_rate=dz.blink_rate, candidates=candidates)
        finished = self.state.update(signals)
        if finished and self.logger is not None:
            self.logger.log_event(finished)
        behavior, conf = self.state.behavior, self.state.confidence
        self.metrics = {
            "behavior": behavior,
            "confidence": conf,
            "calibration_progress": 1.0,
            "ear": ear,
            "blink_rate": dz.blink_rate,
            "perclos": dz.perclos,
            "head_pose": ds.head_pose,
            "yaw": rel[0],
            "pitch": rel[1],
            "fps": self.fps,
            "face_detected": face is not None,
        }
        if behavior in RISKY:
            self.alerts.trigger(behavior)

        if self.logger is not None and st >= self.next_sample:
            self.next_sample += self.cfg["logging"]["sample_every"]
            self.logger.log_sample(st, {
                "Behavior": behavior, "Trigger": self.state.trigger,
                "EAR": _r(ear), "Head_Pose": ds.head_pose, "Yaw": _r(rel[0], 1), "Pitch": _r(rel[1], 1),
                "Roll": _r(pose[2], 1) if pose else None, "Confidence": _r(conf),
                "Alert": "ON" if behavior in RISKY else "OFF", "Blink_Rate": _r(dz.blink_rate, 1),
                "PERCLOS": _r(dz.perclos), "Face_Detected": int(face is not None),
                "Phone_Conf": _r(obj.conf.get("phone", 0)), "Drink_Conf": _r(obj.conf.get("drink", 0)),
                "Food_Conf": _r(obj.conf.get("food", 0)), "True_Label": self.true_label,
                "FPS": _r(self.fps, 1)})

        # ---- drawing ----
        draw_face(frame, face, dz.eyes_closed)
        draw_objects(frame, obj.detections)
        self.alerts.draw_banner(frame, behavior, conf)
        if not getattr(self.args, "dashboard", False):
            lines = [f"State: {behavior} ({conf:.0%})",
                     f"EAR: {ear:.3f} (closed < {self.drowsy.threshold:.3f})" if ear else "EAR: no face",
                     f"Blinks/min: {dz.blink_rate:.0f}   PERCLOS: {dz.perclos:.0%}",
                     f"Head: {ds.head_pose}  yaw {rel[0]:+.0f}  pitch {rel[1]:+.0f}" if pose else "Head: no face",
                     f"{self.session_id}  {self.fps:.0f} FPS" + ("" if self.alerts.audio else "  (muted)")]
            if self.args.test_mode:
                lines.append(f"TRUE LABEL: {self.true_label}  (keys 0-5)")
            draw_panel(frame, lines, behavior)

    def snapshot(self):
        metrics = dict(self.metrics)
        metrics["audio_enabled"] = self.alerts.audio
        metrics["true_label"] = self.true_label
        return metrics

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
            if self.args.test_mode:
                print("Test mode session ended; nothing was saved.")
            else:
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
