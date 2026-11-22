"""Combines all detectors into one driver state, and cuts the timeline into events.

Each frame the detectors may flag several things at once. We pick one state by
priority (most dangerous first):

    Drowsy > Using Phone > Distracted > Drinking > Eating > Normal

An "event" is a stretch of time in one state, e.g. "Using Phone from 21:15:02
to 21:15:20". When the state changes, the finished event is returned so it can
be written to events.csv.
"""
from collections import Counter
from dataclasses import dataclass, field

from . import RISKY

PRIORITY = ["Drowsy", "Using Phone", "Distracted", "Drinking", "Eating"]


@dataclass
class FrameSignals:
    """Everything the detectors said about one frame."""
    t: float
    ear: float = None
    yaw: float = None
    pitch: float = None
    roll: float = None
    head_pose: str = "No Face"
    blink_rate: float = 0.0
    candidates: dict = field(default_factory=dict)  # behavior -> (confidence, trigger)


@dataclass
class Event:
    behavior: str
    start: float
    end: float = None
    trigger: str = ""
    ears: list = field(default_factory=list)
    yaws: list = field(default_factory=list)
    pitches: list = field(default_factory=list)
    rolls: list = field(default_factory=list)
    poses: list = field(default_factory=list)
    confs: list = field(default_factory=list)
    blink_rates: list = field(default_factory=list)

    def add(self, s: FrameSignals, conf):
        if s.ear is not None:
            self.ears.append(s.ear)
        if s.yaw is not None:
            self.yaws.append(s.yaw)
            self.pitches.append(s.pitch)
            self.rolls.append(s.roll)
        self.poses.append(s.head_pose)
        self.confs.append(conf)
        self.blink_rates.append(s.blink_rate)

    @property
    def duration(self):
        return (self.end if self.end is not None else self.start) - self.start

    def summary(self):
        mean = lambda v: round(sum(v) / len(v), 3) if v else None
        return {
            "Behavior": self.behavior,
            "Trigger": self.trigger,
            "Duration_s": round(self.duration, 2),
            "EAR": mean(self.ears),
            "EAR_Min": round(min(self.ears), 3) if self.ears else None,
            "Head_Pose": Counter(self.poses).most_common(1)[0][0] if self.poses else "",
            "Yaw": mean(self.yaws),
            "Pitch": mean(self.pitches),
            "Roll": mean(self.rolls),
            "Confidence": mean(self.confs),
            "Confidence_Max": round(max(self.confs), 3) if self.confs else None,
            "Alert": "ON" if self.behavior in RISKY else "OFF",
            "Blink_Rate": mean(self.blink_rates),
        }


class StateMachine:
    def __init__(self, hold_seconds=0.7):
        self.hold = hold_seconds
        self.behavior = "Normal"
        self.confidence = 1.0
        self.trigger = ""
        self.last_seen = {}       # behavior -> last time it was flagged
        self.last_conf = {}
        self.event = None

    def _choose(self, s: FrameSignals):
        for b, (conf, trig) in s.candidates.items():
            self.last_seen[b] = s.t
            self.last_conf[b] = (conf, trig)
        for b in PRIORITY:
            if b in s.candidates:
                return b, *s.candidates[b]
            # Hold a risky state briefly after its signal stops, to avoid flicker.
            if b == self.behavior and s.t - self.last_seen.get(b, -1e9) <= self.hold:
                return b, *self.last_conf[b]
        return "Normal", _normal_confidence(s), ""

    def update(self, s: FrameSignals):
        """Returns a finished Event when the state changes, otherwise None."""
        behavior, conf, trigger = self._choose(s)
        finished = None
        if self.event is None:
            self.event = Event(behavior, s.t, trigger=trigger)
        elif behavior != self.event.behavior:
            self.event.end = s.t
            finished = self.event
            self.event = Event(behavior, s.t, trigger=trigger)
        self.event.add(s, conf)
        self.behavior, self.confidence, self.trigger = behavior, conf, trigger
        return finished

    def close(self, t):
        """Ends the current event (call when the session stops)."""
        if self.event is None:
            return None
        self.event.end = t
        finished, self.event = self.event, None
        return finished


def _normal_confidence(s: FrameSignals):
    """How sure we are the driver is fine: high when the face is clearly visible and forward."""
    if s.ear is None:
        return 0.5
    return 0.95 if s.head_pose == "Forward" else 0.8
