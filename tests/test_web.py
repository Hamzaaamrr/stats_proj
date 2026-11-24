import csv
import io
import os

import pytest

from monitor.logger import EVENT_FIELDS, SAMPLE_FIELDS, SESSION_FIELDS
from monitor.web import create_app


class FakeWorker:
    def __init__(self, settings):
        self.settings = settings
        self.commands = []
        self.stopped = False

    def start(self):
        pass

    def snapshot(self):
        return {
            "frame": None,
            "frame_version": 0,
            "metrics": {},
            "message": "Running",
            "error": "",
            "finished": self.stopped,
            "session_id": "S01",
        }

    def stop(self):
        self.stopped = True

    def command(self, key):
        self.commands.append(key)


@pytest.fixture
def client(tmp_path):
    workers = []

    def create_worker(settings):
        worker = FakeWorker(settings)
        workers.append(worker)
        return worker

    app = create_app(create_worker, {"driver": "D01"}, data_dir=tmp_path)
    app.testing = True
    with app.test_client() as test_client:
        yield test_client, workers


def valid_settings(**changes):
    settings = {
        "driver": "D01",
        "camera": 0,
        "lighting": "bright",
        "camera_position": "front",
        "glasses": "no",
        "test_mode": True,
        "objects": True,
        "audio": True,
    }
    settings.update(changes)
    return settings


def write_history_file(data_dir, filename, fields, rows):
    with (data_dir / filename).open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def seed_history(data_dir):
    sessions = [
        {
            "Session_ID": "S01",
            "Driver_ID": "D01",
            "Start_Time": "2025-01-01 10:00:00.000",
            "Duration_s": "75.0",
            "Events": "2",
            "Risky_Events": "1",
            "Calibrated": "1",
            "Source": "camera",
            "Lighting": "bright",
            "Camera_Position": "front",
            "Glasses": "no",
        },
        {
            "Session_ID": "S02",
            "Driver_ID": "D02",
            "Start_Time": "2025-01-02 10:00:00.000",
            "Duration_s": "30.0",
            "Events": "1",
            "Risky_Events": "0",
            "Calibrated": "0",
            "Source": "video",
        },
    ]
    events = [
        {
            "Session_ID": session_id,
            "Timestamp": "2025-01-01 10:00:05.000",
            "Duration_s": "5.0",
            "Behavior": behavior,
            "Trigger": "EAR",
            "Alert": alert,
        }
        for session_id, behavior, alert in (
            ("S01", "Drowsy", "ON"),
            ("S02", "Normal", "OFF"),
        )
    ]
    samples = [
        {"Session_ID": session_id, "Driver_ID": driver_id}
        for session_id, driver_id in (("S01", "D01"), ("S02", "D02"))
    ]
    write_history_file(data_dir, "sessions.csv", SESSION_FIELDS, sessions)
    write_history_file(data_dir, "events.csv", EVENT_FIELDS, events)
    write_history_file(data_dir, "samples.csv", SAMPLE_FIELDS, samples)


def test_dashboard_is_served_with_local_privacy_headers(client):
    test_client, _workers = client
    response = test_client.get("/")
    assert response.status_code == 200
    assert b"THIS DEVICE ONLY" in response.data
    assert "frame-ancestors 'none'" in response.headers["Content-Security-Policy"]
    assert test_client.get("/assets/app.css").status_code == 200
    assert test_client.get("/assets/app.js").status_code == 200


def test_web_app_rejects_non_local_hosts(client):
    test_client, _workers = client
    response = test_client.get("/", headers={"Host": "remote.example"})
    assert response.status_code == 403


def test_start_validates_settings_and_reports_running_status(client):
    test_client, workers = client
    response = test_client.post("/api/start", json=valid_settings())
    assert response.status_code == 202
    assert workers[0].settings["driver"] == "D01"
    assert test_client.get("/api/status").json["running"] is True


def test_start_rejects_invalid_camera_index(client):
    test_client, workers = client
    response = test_client.post("/api/start", json=valid_settings(camera=-1))
    assert response.status_code == 400
    assert not workers


def test_cli_video_is_used_when_no_upload_is_selected():
    workers = []

    def create_worker(settings):
        worker = FakeWorker(settings)
        workers.append(worker)
        return worker

    app = create_app(create_worker, initial_video="C:\\clips\\drive.mp4")
    app.testing = True
    response = app.test_client().post("/api/start", json=valid_settings())
    assert response.status_code == 202
    assert workers[0].settings["video"] == "C:\\clips\\drive.mp4"


def test_duplicate_start_is_rejected(client):
    test_client, _workers = client
    test_client.post("/api/start", json=valid_settings())
    response = test_client.post("/api/start", json=valid_settings())
    assert response.status_code == 409


def test_commands_are_limited_to_supported_controls(client):
    test_client, workers = client
    test_client.post("/api/start", json=valid_settings())
    response = test_client.post("/api/command", json={
        "action": "true_label", "label": "Using Phone",
    })
    assert response.status_code == 200
    assert workers[0].commands == [ord("3")]

    invalid = test_client.post("/api/command", json={"action": "arbitrary"})
    assert invalid.status_code == 400


def test_cross_origin_session_starts_are_rejected(client):
    test_client, workers = client
    response = test_client.post(
        "/api/start",
        json=valid_settings(),
        headers={"Origin": "https://untrusted.example"},
    )
    assert response.status_code == 403
    assert not workers


def test_video_upload_rejects_unexpected_file_types(client):
    test_client, _workers = client
    response = test_client.post(
        "/api/upload-video",
        data={"video": (io.BytesIO(b"not a video"), "payload.exe")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 400


def test_video_upload_is_passed_to_the_worker_and_cleaned_up(client):
    test_client, workers = client
    uploaded = test_client.post(
        "/api/upload-video",
        data={"video": (io.BytesIO(b"local test video"), "clip.mp4")},
        content_type="multipart/form-data",
    )
    assert uploaded.status_code == 200

    started = test_client.post(
        "/api/start",
        json=valid_settings(video_upload=uploaded.json["upload_id"]),
    )
    assert started.status_code == 202
    path = workers[0].settings["video"]
    assert os.path.isfile(path)
    assert workers[0].settings["temporary_video"] == path
    os.remove(path)


def test_history_lists_pages_and_returns_session_events(client, tmp_path):
    test_client, _workers = client
    seed_history(tmp_path)

    page = test_client.get("/api/history?offset=0&limit=1")
    assert page.status_code == 200
    assert page.json["total"] == 2
    assert [row["Session_ID"] for row in page.json["items"]] == ["S02"]
    next_page = test_client.get("/api/history?offset=1&limit=1")
    assert [row["Session_ID"] for row in next_page.json["items"]] == ["S01"]

    detail = test_client.get("/api/history/S01")
    assert detail.status_code == 200
    assert detail.json["session"]["Driver_ID"] == "D01"
    assert detail.json["events"][0]["Behavior"] == "Drowsy"


def test_history_delete_removes_selected_session_from_all_csvs(client, tmp_path):
    test_client, _workers = client
    seed_history(tmp_path)

    response = test_client.delete("/api/history", json={"session_ids": ["S01"]})
    assert response.status_code == 200
    assert response.json["deleted_sessions"] == ["S01"]
    for filename in ("sessions.csv", "events.csv", "samples.csv"):
        with (tmp_path / filename).open(newline="", encoding="utf-8") as source:
            rows = list(csv.DictReader(source))
        assert [row["Session_ID"] for row in rows] == ["S02"]


def test_history_delete_rejects_active_session_and_invalid_ids(client, tmp_path):
    test_client, _workers = client
    seed_history(tmp_path)
    test_client.post("/api/start", json=valid_settings())

    active = test_client.delete("/api/history", json={"session_ids": ["S01"]})
    assert active.status_code == 409

    invalid = test_client.delete("/api/history", json={"session_ids": []})
    assert invalid.status_code == 400
    test_client.post("/api/stop")
    unknown = test_client.delete("/api/history", json={"session_ids": ["S99"]})
    assert unknown.status_code == 404


def test_history_detail_reports_unknown_session(client):
    test_client, _workers = client
    response = test_client.get("/api/history/unknown")
    assert response.status_code == 404
