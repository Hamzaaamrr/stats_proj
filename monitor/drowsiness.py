"""Drowsiness from EAR over time: long eye closures, PERCLOS and blink rate.

Three warning signs, any of which makes the driver Drowsy:
  1. Microsleep: eyes closed for >= closed_seconds (1.5 s) in one go.
  2. PERCLOS: eyes closed for more than 30% of the last 60 seconds.
  3. Blink rate: more than 25 blinks per minute (a sign of fatigue).
A blink is a closure shorter than blink_max_seconds (0.5 s).
"""
from collections import deque
from dataclasses import dataclass


@dataclass
class DrowsyStatus:
    eyes_closed: bool
    closed_for: float     # seconds the current closure has lasted
    blink_rate: float     # blinks per minute
    perclos: float        # 0..1
    drowsy: bool
    confidence: float
    trigger: str          # "eye_closure", "perclos", "blink_rate" or ""


class DrowsinessTracker:
    def __init__(self, cfg, ear_threshold):
        self.cfg = cfg
        self.threshold = ear_threshold
        self.closed_since = None
        self.blinks = deque()    # times of recent blinks
        self.history = deque()   # (t, closed) for PERCLOS
        self.t_start = None

    def update(self, t, ear):
        """Call once per frame. ear=None when no face is visible."""
        cfg = self.cfg
        if self.t_start is None:
            self.t_start = t
        closed = ear is not None and ear < self.threshold

        # Track closures and count blinks when the eye reopens.
        # If the face is lost for a moment we leave the closure state as it is.
        if ear is None:
            pass
        elif closed and self.closed_since is None:
            self.closed_since = t
        elif not closed and self.closed_since is not None:
            if t - self.closed_since <= cfg["blink_max_seconds"]:
                self.blinks.append(t)
            self.closed_since = None
        closed_for = t - self.closed_since if self.closed_since is not None else 0.0

        # Blink rate over the last minute.
        while self.blinks and t - self.blinks[0] > 60:
            self.blinks.popleft()
        elapsed = min(60.0, t - self.t_start)
        blink_rate = len(self.blinks) * 60.0 / elapsed if elapsed >= 1 else 0.0

        # PERCLOS over the window (only frames where a face was seen).
        if ear is not None:
            self.history.append((t, closed))
        while self.history and t - self.history[0][0] > cfg["perclos_window"]:
            self.history.popleft()
        perclos = (sum(c for _, c in self.history) / len(self.history)) if self.history else 0.0

        drowsy, conf, trigger = False, 0.0, ""
        if closed_for >= cfg["closed_seconds"]:
            depth = (self.threshold - ear) / (0.5 * self.threshold) if ear is not None else 1.0
            extra = (closed_for - cfg["closed_seconds"]) / cfg["closed_seconds"]
            conf = 0.6 + 0.2 * _clip(depth) + 0.2 * _clip(extra)
            drowsy, trigger = True, "eye_closure"
        elif elapsed >= cfg["perclos_window"] / 2 and perclos >= cfg["perclos_threshold"]:
            conf = 0.5 + 0.5 * _clip((perclos - cfg["perclos_threshold"]) / cfg["perclos_threshold"])
            drowsy, trigger = True, "perclos"
        elif elapsed >= cfg["blink_rate_min_window"] and blink_rate > cfg["blink_rate_max"]:
            conf = 0.5 + 0.3 * _clip((blink_rate - cfg["blink_rate_max"]) / cfg["blink_rate_max"])
            drowsy, trigger = True, "blink_rate"

        return DrowsyStatus(closed, closed_for, blink_rate, perclos, drowsy, round(conf, 3), trigger)


def _clip(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))
