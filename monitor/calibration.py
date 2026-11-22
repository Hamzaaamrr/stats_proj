"""Calibration: learn this driver's normal open-eye EAR and forward head pose.

Everyone's eyes are a different shape, and the camera may sit at an angle. For
the first few seconds the driver looks at the road with eyes open; we take the
median of what we see and measure everything else against it.
"""
from statistics import median


class Calibrator:
    def __init__(self, seconds, ear_ratio, ear_fallback):
        self.seconds = seconds
        self.ear_ratio = ear_ratio
        self.ear_fallback = ear_fallback
        self.ears, self.yaws, self.pitches = [], [], []
        self.t_start = None
        self.done = False

    def add(self, t, ear, pose):
        if self.t_start is None:
            self.t_start = t
        if ear is not None and pose is not None:
            self.ears.append(ear)
            self.yaws.append(pose[0])
            self.pitches.append(pose[1])
        if t - self.t_start >= self.seconds:
            self.done = True
        return self.done

    def progress(self, t):
        return 0.0 if self.t_start is None else min(1.0, (t - self.t_start) / self.seconds)

    def result(self):
        """Returns dict with baseline values. Falls back to defaults if the face was rarely seen."""
        if len(self.ears) < 10:
            return {"ear_baseline": None, "ear_threshold": self.ear_fallback,
                    "yaw0": 0.0, "pitch0": 0.0, "ok": False}
        ear0 = median(self.ears)
        return {"ear_baseline": round(ear0, 4),
                "ear_threshold": round(ear0 * self.ear_ratio, 4),
                "yaw0": round(median(self.yaws), 2),
                "pitch0": round(median(self.pitches), 2),
                "ok": True}
