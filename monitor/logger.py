"""Writes the three CSV files used for the statistical analysis in R.

data/events.csv   one row per event (a stretch of time in one state)
data/samples.csv  one row per second: predicted state + the true label in test mode
data/sessions.csv one row per session: who, when, lighting, camera position, ...

All files are appended to, so every session ends up in the same three files and
is told apart by Session_ID.
"""
import csv
import os
from datetime import datetime, timedelta

EVENT_FIELDS = ["Session_ID", "Event_ID", "Driver_ID", "Timestamp", "End_Time", "Duration_s",
                "Behavior", "Trigger", "EAR", "EAR_Min", "Head_Pose", "Yaw", "Pitch", "Roll",
                "Confidence", "Confidence_Max", "Alert", "Blink_Rate"]
SAMPLE_FIELDS = ["Session_ID", "Driver_ID", "Timestamp", "Time_s", "Behavior", "Trigger", "EAR",
                 "Head_Pose", "Yaw", "Pitch", "Roll", "Confidence", "Alert", "Blink_Rate",
                 "PERCLOS", "Face_Detected", "Phone_Conf", "Drink_Conf", "Food_Conf",
                 "True_Label", "FPS"]
SESSION_FIELDS = ["Session_ID", "Driver_ID", "Start_Time", "End_Time", "Duration_s", "Lighting",
                  "Camera_Position", "Glasses", "Test_Mode", "Source", "Calibrated",
                  "EAR_Baseline", "EAR_Threshold", "Yaw_Baseline", "Pitch_Baseline",
                  "Events", "Risky_Events", "Notes"]

TIME_FMT = "%Y-%m-%d %H:%M:%S.%f"


def next_session_id(data_dir):
    """S01, S02, ... based on sessions already in sessions.csv."""
    path = os.path.join(data_dir, "sessions.csv")
    n = 0
    if os.path.exists(path):
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                sid = row.get("Session_ID", "")
                if sid[:1] == "S" and sid[1:].isdigit():
                    n = max(n, int(sid[1:]))
    return f"S{n + 1:02d}"


class SessionLogger:
    def __init__(self, data_dir, session_id, driver_id, start_wallclock=None):
        os.makedirs(data_dir, exist_ok=True)
        self.data_dir = data_dir
        self.session_id = session_id
        self.driver_id = driver_id
        self.start = start_wallclock or datetime.now()
        self.event_no = 0
        self.risky_events = 0
        self._files = {}

    def stamp(self, t):
        """Wall-clock time string for t seconds after the session started."""
        return (self.start + timedelta(seconds=t)).strftime(TIME_FMT)[:-3]

    def _writer(self, name, fields):
        if name not in self._files:
            path = os.path.join(self.data_dir, name)
            new = not os.path.exists(path) or os.path.getsize(path) == 0
            f = open(path, "a", newline="", encoding="utf-8")
            w = csv.DictWriter(f, fieldnames=fields)
            if new:
                w.writeheader()
            self._files[name] = (f, w)
        return self._files[name][1]

    def log_event(self, event):
        self.event_no += 1
        row = event.summary()
        if row["Alert"] == "ON":
            self.risky_events += 1
        row.update({"Session_ID": self.session_id, "Event_ID": self.event_no,
                    "Driver_ID": self.driver_id, "Timestamp": self.stamp(event.start),
                    "End_Time": self.stamp(event.end)})
        self._writer("events.csv", EVENT_FIELDS).writerow(row)

    def log_sample(self, t, values):
        row = {"Session_ID": self.session_id, "Driver_ID": self.driver_id,
               "Timestamp": self.stamp(t), "Time_s": round(t, 2)}
        row.update(values)
        self._writer("samples.csv", SAMPLE_FIELDS).writerow(row)

    def log_session(self, end_t, info):
        row = {"Session_ID": self.session_id, "Driver_ID": self.driver_id,
               "Start_Time": self.stamp(0), "End_Time": self.stamp(end_t),
               "Duration_s": round(end_t, 1), "Events": self.event_no,
               "Risky_Events": self.risky_events}
        row.update(info)
        self._writer("sessions.csv", SESSION_FIELDS).writerow(row)

    def flush(self):
        for f, _ in self._files.values():
            f.flush()

    def close(self):
        for f, _ in self._files.values():
            f.close()
        self._files = {}
