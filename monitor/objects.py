"""Phone, drinking and eating detection with a pretrained YOLO model.

YOLO (trained on the COCO dataset) already recognises 80 everyday objects,
including "cell phone", "cup", "bottle" and many foods. We group them:

    phone -> Using Phone   (anywhere in view: at the ear or texting in the lap)
    drink -> Drinking      (cup/bottle must be near the face)
    food  -> Eating        (food must be near the face)

"Near the face" stops a cup sitting in the cup holder from counting. To avoid
reacting to a single wrong frame, an object must show up in most of the recent
YOLO runs (e.g. 60% of the last 15) before it counts.
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
    detections: list = field(default_factory=list)


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
        self.min_conf = min(cfg["conf"].values())

    def detect(self, frame):
        res = self.model.predict(frame, imgsz=self.cfg["imgsz"], conf=self.min_conf,
                                 classes=self.class_ids, verbose=False)[0]
        out = []
        for box, cls, conf in zip(res.boxes.xyxy.tolist(), res.boxes.cls.tolist(),
                                  res.boxes.conf.tolist()):
            name = self.model.names[int(cls)]
            group = self.name_to_group[name]
            if conf >= self.cfg["conf"][group]:
                out.append(Detection(name, group, float(conf), tuple(box)))
        return out


class ObjectTracker:
    """Smooths YOLO results over the last few runs and applies the near-face rule."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.history = {g: deque(maxlen=cfg["window"]) for g in GROUP_TO_BEHAVIOR}
        self.last = ObjectStatus()

    def update(self, detections, face_bbox):
        """Call after each YOLO run. face_bbox may be None if no face is visible."""
        near = expand_box(face_bbox, self.cfg["face_margin"]) if face_bbox else None
        best = {g: 0.0 for g in GROUP_TO_BEHAVIOR}
        kept = []
        for d in detections:
            if d.group in NEEDS_FACE and (near is None or not boxes_overlap(d.box, near)):
                continue
            kept.append(d)
            best[d.group] = max(best[d.group], d.conf)
        status = ObjectStatus(detections=kept, conf=dict(best))
        for g, hist in self.history.items():
            hist.append(best[g])
            hits = [c for c in hist if c > 0]
            if len(hist) >= self.cfg["window"] // 2 and len(hits) / len(hist) >= self.cfg["min_fraction"]:
                status.active[g] = round(sum(hits) / len(hits), 3)
        self.last = status
        return status
