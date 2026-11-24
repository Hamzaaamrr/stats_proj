"""Phone, drinking and eating detection with a pretrained YOLO model.

YOLO (trained on the COCO dataset) already recognises 80 everyday objects,
including "cell phone", "cup", "bottle" and many foods. We group them:

    phone -> Using Phone   (anywhere in view: at the ear or texting in the lap)
    drink -> Drinking      (cup/bottle must be near the face)
    food  -> Eating        (food must be near the face)

"Near the face" stops a cup sitting in the cup holder from counting. To avoid
reacting to a single wrong frame, an object must show up in enough of the recent
YOLO runs (e.g. 40% of the last 15) before it counts.

Two extra rules for objects that hide the face:
  * Face memory: a bottle held in front of the face can make the face tracker
    lose the face. We then use where the face was a moment ago (up to
    face_memory_seconds) instead of throwing the bottle away.
  * "At face": the time an object was last seen at the face (or any object
    group was active). main.py uses it to ignore eye closure / no-face while
    drinking or on the phone, so they aren't mislabelled Drowsy/Distracted.
"""
from collections import deque
from dataclasses import dataclass, field

GROUP_TO_BEHAVIOR = {"phone": "Using Phone", "drink": "Drinking", "food": "Eating"}
NEEDS_FACE = {"drink", "food"}


@dataclass
class Detection:
    name: str
    group: str
    conf: float
    box: tuple  # (x1, y1, x2, y2)


@dataclass
class ObjectStatus:
    active: dict = field(default_factory=dict)  # group -> confidence, for groups that pass
    conf: dict = field(default_factory=dict)    # group -> latest best confidence (for logging)
    raw: dict = field(default_factory=dict)     # group -> best confidence before any filtering
    detections: list = field(default_factory=list)
    at_face: bool = False                       # an object is at the face in this run


def expand_box(box, margin):
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    return (x1 - w * margin, y1 - h * margin, x2 + w * margin, y2 + h * margin)


def boxes_overlap(a, b):
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


class ObjectDetector:
    """Wraps YOLO and keeps only the classes we care about."""

    def __init__(self, cfg):
        from ultralytics import YOLO  # imported here so tests can run without loading torch
        self.cfg = cfg
        self.model = YOLO(cfg["model"])
        self.name_to_group = {n: g for g, names in cfg["classes"].items() for n in names}
        self.class_ids = [i for i, n in self.model.names.items() if n in self.name_to_group]
        # Ask YOLO for weak results too, so they can be logged; the per-group
        # thresholds are applied in ObjectTracker.
        self.min_conf = min([cfg.get("raw_floor", 0.10)] + list(cfg["conf"].values()))

    def detect(self, frame):
        res = self.model.predict(frame, imgsz=self.cfg["imgsz"], conf=self.min_conf,
                                 classes=self.class_ids, verbose=False)[0]
        out = []
        for box, cls, conf in zip(res.boxes.xyxy.tolist(), res.boxes.cls.tolist(),
                                  res.boxes.conf.tolist()):
            name = self.model.names[int(cls)]
            out.append(Detection(name, self.name_to_group[name], float(conf), tuple(box)))
        return out


class ObjectTracker:
    """Smooths YOLO results over the last few runs and applies the near-face rule."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.history = {g: deque(maxlen=cfg["window"]) for g in GROUP_TO_BEHAVIOR}
        self.last = ObjectStatus()
        self.last_face = None          # (time, bbox) of the last face we saw
        self.last_at_face_t = None     # last time an object was at the face

    def _face_box(self, face_bbox, t):
        """The face box, or the last one seen if the face was lost only briefly."""
        if face_bbox:
            self.last_face = (t, face_bbox)
            return face_bbox
        if self.last_face and t is not None and t - self.last_face[0] <= self.cfg.get("face_memory_seconds", 0):
            return self.last_face[1]
        return None

    def update(self, detections, face_bbox, t=None):
        """Call after each YOLO run. face_bbox may be None if no face is visible;
        t (seconds) enables the face memory and the at-face timer."""
        face_box = self._face_box(face_bbox, t)
        near = expand_box(face_box, self.cfg["face_margin"]) if face_box else None
        best = {g: 0.0 for g in GROUP_TO_BEHAVIOR}
        raw = {g: 0.0 for g in GROUP_TO_BEHAVIOR}
        kept, at_face = [], False
        for d in detections:
            raw[d.group] = max(raw[d.group], d.conf)
            if d.conf < self.cfg["conf"][d.group]:
                continue
            is_near = near is not None and boxes_overlap(d.box, near)
            if d.group in NEEDS_FACE and not is_near:
                continue
            kept.append(d)
            at_face = at_face or is_near
            best[d.group] = max(best[d.group], d.conf)
        status = ObjectStatus(detections=kept, conf=dict(best), raw=raw)
        for g, hist in self.history.items():
            hist.append(best[g])
            hits = [c for c in hist if c > 0]
            if len(hist) >= self.cfg["window"] // 2 and len(hits) / len(hist) >= self.cfg["min_fraction"]:
                status.active[g] = round(sum(hits) / len(hits), 3)
        status.at_face = at_face or bool(status.active)
        if status.at_face and t is not None:
            self.last_at_face_t = t
        self.last = status
        return status

    def object_at_face_recently(self, t):
        hold = self.cfg.get("at_face_hold_seconds", 0)
        return self.last_at_face_t is not None and t - self.last_at_face_t <= hold
