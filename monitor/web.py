"""Loopback-only web dashboard and local session API."""
import csv
import os
import shutil
import tempfile
import threading
import time
import uuid
from pathlib import Path
from urllib.parse import urlparse

import cv2
from flask import Flask, Response, abort, jsonify, render_template, request
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.utils import secure_filename

from monitor.logger import EVENT_FIELDS, SAMPLE_FIELDS, SESSION_FIELDS


ALLOWED_VIDEO_EXTENSIONS = {".avi", ".mkv", ".mov", ".mp4"}
TRUE_LABELS = ("Normal", "Drowsy", "Distracted", "Using Phone", "Eating", "Drinking")
HISTORY_FILES = (
    ("sessions.csv", SESSION_FIELDS),
    ("events.csv", EVENT_FIELDS),
    ("samples.csv", SAMPLE_FIELDS),
)


class SessionHistory:
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.lock = threading.RLock()

    def list_sessions(self, offset=0, limit=25):
        with self.lock:
            _fields, rows = self._read_csv("sessions.csv", SESSION_FIELDS)
            rows = [row for row in rows if row.get("Session_ID")]
            rows.sort(key=lambda row: row.get("Start_Time", ""), reverse=True)
            return {"items": rows[offset:offset + limit], "total": len(rows)}

    def get_session(self, session_id):
        with self.lock:
            _fields, sessions = self._read_csv("sessions.csv", SESSION_FIELDS)
            session = next(
                (row for row in sessions if row.get("Session_ID") == session_id),
                None,
            )
            if session is None:
                raise KeyError(session_id)
            _fields, events = self._read_csv("events.csv", EVENT_FIELDS)
            events = [row for row in events if row.get("Session_ID") == session_id]
            return {"session": session, "events": events}

    def delete_sessions(self, session_ids):
        selected = set(session_ids)
        if not selected:
            raise ValueError("Select at least one session to delete.")

        with self.lock:
            _fields, sessions = self._read_csv("sessions.csv", SESSION_FIELDS)
            existing = {row.get("Session_ID") for row in sessions if row.get("Session_ID")}
            missing = selected - existing
            if missing:
                raise KeyError(", ".join(sorted(missing)))

            staged = {}
            deleted_counts = {}
            for filename, default_fields in HISTORY_FILES:
                path = self.data_dir / filename
                if not path.exists():
                    continue
                fields, rows = self._read_csv(filename, default_fields)
                retained = [row for row in rows if row.get("Session_ID") not in selected]
                deleted_counts[filename] = len(rows) - len(retained)
                staged[path] = self._stage_csv(path, fields, retained)

            self._replace_csvs(staged)
            return {
                "deleted_sessions": sorted(selected),
                "deleted_counts": deleted_counts,
            }

    def _read_csv(self, filename, default_fields):
        path = self.data_dir / filename
        if not path.exists() or path.stat().st_size == 0:
            return list(default_fields), []
        with path.open("r", newline="", encoding="utf-8-sig") as source:
            reader = csv.DictReader(source)
            if reader.fieldnames is None or "Session_ID" not in reader.fieldnames:
                raise RuntimeError(f"History file has an invalid header: {path}")
            return reader.fieldnames, list(reader)

    def _stage_csv(self, path, fields, rows):
        descriptor, temporary = tempfile.mkstemp(
            prefix=".history-", suffix=".tmp", dir=str(self.data_dir))
        try:
            with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as destination:
                writer = csv.DictWriter(destination, fieldnames=fields, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(rows)
        except Exception:
            try:
                os.remove(temporary)
            except FileNotFoundError:
                pass
            raise
        return temporary

    def _replace_csvs(self, staged):
        backups = {}
        replaced = []
        preserve_backups = set()
        try:
            for path in staged:
                descriptor, backup = tempfile.mkstemp(
                    prefix=".history-backup-", suffix=".tmp", dir=str(self.data_dir))
                os.close(descriptor)
                backups[path] = backup
                shutil.copy2(path, backup)

            for path, temporary in staged.items():
                os.replace(temporary, path)
                replaced.append(path)
        except Exception as exc:
            rollback_errors = []
            for path in reversed(replaced):
                try:
                    os.replace(backups[path], path)
                    backups.pop(path, None)
                except Exception as rollback_exc:
                    preserve_backups.add(backups[path])
                    rollback_errors.append(rollback_exc)
            if rollback_errors:
                raise RuntimeError(
                    "Session deletion failed and one or more CSV files could not be restored. "
                    f"Recovery copies remain at: {', '.join(sorted(preserve_backups))}"
                ) from rollback_errors[0]
            raise RuntimeError(
                "Session deletion failed; the original CSV files were restored."
            ) from exc
        finally:
            temporary_files = list(staged.values()) + [
                backup for backup in backups.values() if backup not in preserve_backups
            ]
            for temporary in temporary_files:
                try:
                    os.remove(temporary)
                except FileNotFoundError:
                    pass


class WebSessionController:
    def __init__(self, worker_factory, data_dir, initial_video=""):
        self.worker_factory = worker_factory
        self.initial_video = initial_video
        self.lock = threading.Lock()
        self.worker = None
        self.uploads = {}
        self.history = SessionHistory(data_dir)

    def add_video(self, uploaded_file):
        filename = secure_filename(uploaded_file.filename or "")
        extension = os.path.splitext(filename)[1].lower()
        if not filename or extension not in ALLOWED_VIDEO_EXTENSIONS:
            raise ValueError("Choose an AVI, MKV, MOV, or MP4 video.")

        descriptor, path = tempfile.mkstemp(prefix="driver-monitor-", suffix=extension)
        os.close(descriptor)
        try:
            uploaded_file.save(path)
        except Exception:
            os.remove(path)
            raise

        with self.lock:
            self._purge_expired_uploads()
            upload_id = uuid.uuid4().hex
            self.uploads[upload_id] = (path, time.monotonic())
        return upload_id

    def start(self, settings):
        with self.lock:
            self._purge_expired_uploads()
            if self.worker is not None and not self.worker.snapshot()["finished"]:
                raise RuntimeError("A monitoring session is already running.")

            upload_id = settings.pop("video_upload", "")
            if upload_id:
                upload = self.uploads.pop(upload_id, None)
                if upload is None:
                    raise ValueError("The uploaded video expired. Please select it again.")
                settings["video"] = upload[0]
                settings["temporary_video"] = upload[0]
            elif self.initial_video:
                settings["video"] = self.initial_video

            try:
                worker = self.worker_factory(settings)
                self.worker = worker
                worker.start()
            except Exception:
                self.worker = None
                temporary_video = settings.get("temporary_video")
                if temporary_video:
                    try:
                        os.remove(temporary_video)
                    except FileNotFoundError:
                        pass
                raise
        return worker.snapshot()

    def stop(self):
        with self.lock:
            if self.worker is None or self.worker.snapshot()["finished"]:
                raise RuntimeError("There is no active monitoring session.")
            self.worker.stop()

    def command(self, action, label=""):
        keys = {"recalibrate": ord("c"), "toggle_audio": ord("m")}
        if action == "true_label":
            if label not in TRUE_LABELS:
                raise ValueError("Choose a valid true label.")
            key = ord("0") + TRUE_LABELS.index(label)
        elif action in keys:
            key = keys[action]
        else:
            raise ValueError("Unknown dashboard action.")

        with self.lock:
            if self.worker is None or self.worker.snapshot()["finished"]:
                raise RuntimeError("There is no active monitoring session.")
            self.worker.command(key)

    def delete_sessions(self, session_ids):
        with self.lock:
            if self.worker is not None and not self.worker.snapshot()["finished"]:
                raise RuntimeError("Stop the active monitoring session before deleting history.")
            return self.history.delete_sessions(session_ids)

    def snapshot(self):
        with self.lock:
            worker = self.worker
            if worker is None:
                return {
                    "running": False,
                    "finished": False,
                    "message": "Ready to start monitoring.",
                    "error": "",
                    "session_id": "",
                    "saved": False,
                    "metrics": {},
                    "has_frame": False,
                    "frame": None,
                    "frame_version": 0,
                }
            result = worker.snapshot()

        result.update({
            "running": not result["finished"],
            "has_frame": result["frame"] is not None,
        })
        return result

    def _purge_expired_uploads(self):
        cutoff = time.monotonic() - 3600
        expired = [key for key, (_path, created) in self.uploads.items() if created < cutoff]
        for key in expired:
            path, _created = self.uploads.pop(key)
            try:
                os.remove(path)
            except FileNotFoundError:
                pass


def _validated_settings(data):
    if not isinstance(data, dict):
        raise ValueError("Session settings must be provided as JSON.")

    driver = str(data.get("driver", "")).strip()
    if not driver or len(driver) > 40:
        raise ValueError("Driver ID must contain between 1 and 40 characters.")

    camera = data.get("camera", 0)
    if isinstance(camera, bool):
        raise ValueError("Camera index must be a non-negative integer.")
    try:
        camera = int(camera)
    except (TypeError, ValueError) as exc:
        raise ValueError("Camera index must be a non-negative integer.") from exc
    if camera < 0:
        raise ValueError("Camera index must be a non-negative integer.")

    lighting = str(data.get("lighting", ""))
    position = str(data.get("camera_position", ""))
    glasses = str(data.get("glasses", ""))
    if lighting not in ("", "bright", "dim", "night"):
        raise ValueError("Choose a valid lighting option.")
    if position not in ("", "front", "left-45", "right-45"):
        raise ValueError("Choose a valid camera position.")
    if glasses not in ("", "yes", "no"):
        raise ValueError("Choose a valid glasses option.")

    return {
        "driver": driver,
        "camera": camera,
        "video": "",
        "video_upload": str(data.get("video_upload", "")),
        "lighting": lighting,
        "camera_position": position,
        "glasses": glasses,
        "test_mode": data.get("test_mode") is True,
        "objects": data.get("objects") is True,
        "audio": data.get("audio") is True,
    }


def create_app(worker_factory, defaults=None, initial_video="", data_dir=None):
    web_dir = os.path.join(os.path.dirname(__file__), "web_ui")
    app = Flask(
        __name__,
        template_folder=web_dir,
        static_folder=web_dir,
        static_url_path="/assets",
    )
    app.config["MAX_CONTENT_LENGTH"] = 512 * 1024 * 1024
    if data_dir is None:
        data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    controller = WebSessionController(worker_factory, data_dir, initial_video)
    app.extensions["monitor_controller"] = controller
    defaults = defaults or {}

    @app.after_request
    def add_security_headers(response):
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; img-src 'self' data:; style-src 'self'; "
            "script-src 'self'; connect-src 'self'; object-src 'none'; "
            "base-uri 'self'; frame-ancestors 'none'; form-action 'self'"
        )
        return response

    @app.errorhandler(RequestEntityTooLarge)
    def file_too_large(_error):
        return jsonify(error="Video upload is too large (maximum 512 MB)."), 413

    @app.before_request
    def allow_loopback_requests_only():
        host = request.host.rsplit(":", 1)[0].strip("[]").lower()
        if host not in ("127.0.0.1", "localhost"):
            abort(403)

        origin = request.headers.get("Origin")
        if request.method in ("POST", "PUT", "PATCH", "DELETE") and origin:
            parsed = urlparse(origin)
            if parsed.scheme != "http" or parsed.hostname not in ("127.0.0.1", "localhost"):
                abort(403)

    @app.get("/")
    def index():
        return render_template("index.html", defaults=defaults)

    @app.get("/api/status")
    def status():
        current = controller.snapshot()
        current.pop("frame", None)
        current.pop("frame_version", None)
        return jsonify(current)

    @app.get("/api/history")
    def history():
        try:
            limit = int(request.args.get("limit", 25))
            offset = int(request.args.get("offset", 0))
        except ValueError:
            return jsonify(error="History page must use numeric offset and limit values."), 400
        if not 1 <= limit <= 100 or offset < 0:
            return jsonify(error="History page is out of range."), 400
        return jsonify(controller.history.list_sessions(offset, limit))

    @app.get("/api/history/<session_id>")
    def session_detail(session_id):
        try:
            return jsonify(controller.history.get_session(session_id))
        except KeyError:
            return jsonify(error="Session was not found."), 404

    @app.delete("/api/history")
    def delete_history():
        data = request.get_json(silent=True)
        if not isinstance(data, dict) or not isinstance(data.get("session_ids"), list):
            return jsonify(error="Provide session_ids as a JSON list."), 400
        session_ids = data["session_ids"]
        if (not session_ids or len(session_ids) > 100
                or any(not isinstance(value, str) or not value.strip() for value in session_ids)):
            return jsonify(error="Select between 1 and 100 valid sessions."), 400
        try:
            result = controller.delete_sessions(session_ids)
        except ValueError as exc:
            return jsonify(error=str(exc)), 400
        except KeyError:
            return jsonify(error="One or more selected sessions no longer exist."), 404
        except RuntimeError as exc:
            return jsonify(error=str(exc)), 409
        return jsonify(result)

    @app.post("/api/upload-video")
    def upload_video():
        video = request.files.get("video")
        if video is None:
            return jsonify(error="Choose a video to upload."), 400
        try:
            upload_id = controller.add_video(video)
        except ValueError as exc:
            return jsonify(error=str(exc)), 400
        return jsonify(upload_id=upload_id)

    @app.post("/api/start")
    def start():
        try:
            settings = _validated_settings(request.get_json(silent=True))
            state = controller.start(settings)
        except ValueError as exc:
            return jsonify(error=str(exc)), 400
        except RuntimeError as exc:
            return jsonify(error=str(exc)), 409
        state.pop("frame", None)
        state.pop("frame_version", None)
        return jsonify(state), 202

    @app.post("/api/stop")
    def stop():
        try:
            controller.stop()
        except RuntimeError as exc:
            return jsonify(error=str(exc)), 409
        return jsonify(message="Stopping session and saving data.")

    @app.post("/api/command")
    def command():
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return jsonify(error="Command must be provided as JSON."), 400
        try:
            controller.command(data.get("action", ""), data.get("label", ""))
        except ValueError as exc:
            return jsonify(error=str(exc)), 400
        except RuntimeError as exc:
            return jsonify(error=str(exc)), 409
        return jsonify(message="Command accepted.")

    @app.get("/video-feed")
    def video_feed():
        def generate():
            last_version = -1
            while True:
                current = controller.snapshot()
                frame = current["frame"]
                version = current["frame_version"]
                if frame is not None and version != last_version:
                    encoded, jpeg = cv2.imencode(
                        ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 78])
                    if encoded:
                        last_version = version
                        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                               + jpeg.tobytes() + b"\r\n")
                if current["finished"]:
                    break
                time.sleep(0.06)

        return Response(
            generate(),
            mimetype="multipart/x-mixed-replace; boundary=frame",
            headers={"X-Accel-Buffering": "no"},
        )

    return app
