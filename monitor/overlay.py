"""Draws the live view: eye points, face box, YOLO boxes and an info panel."""
import cv2

from .face import LEFT_EYE, RIGHT_EYE

GREEN, RED, YELLOW, WHITE, GREY = (0, 200, 0), (0, 0, 255), (0, 220, 255), (255, 255, 255), (60, 60, 60)
GROUP_COLORS = {"phone": (255, 0, 255), "drink": (255, 160, 0), "food": (0, 165, 255)}


def put(frame, text, org, color=WHITE, scale=0.55, thick=1):
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def draw_face(frame, face, eyes_closed):
    if face is None:
        return
    color = RED if eyes_closed else GREEN
    for i in LEFT_EYE + RIGHT_EYE:
        x, y = face.landmarks[i]
        cv2.circle(frame, (int(x), int(y)), 2, color, -1)
    x1, y1, x2, y2 = map(int, face.bbox)
    cv2.rectangle(frame, (x1, y1), (x2, y2), GREY, 1)


def draw_objects(frame, detections):
    for d in detections:
        x1, y1, x2, y2 = map(int, d.box)
        c = GROUP_COLORS.get(d.group, YELLOW)
        cv2.rectangle(frame, (x1, y1), (x2, y2), c, 2)
        put(frame, f"{d.name} {d.conf:.2f}", (x1, max(15, y1 - 6)), c)


def draw_panel(frame, lines, behavior):
    """Info lines in the bottom-left corner."""
    h = frame.shape[0]
    y = h - 12 - 22 * (len(lines) - 1)
    for i, line in enumerate(lines):
        color = (RED if behavior != "Normal" else GREEN) if i == 0 else WHITE
        put(frame, line, (10, y), color, 0.6 if i == 0 else 0.5, 2 if i == 0 else 1)
        y += 22


def draw_calibration(frame, progress):
    h, w = frame.shape[:2]
    put(frame, "CALIBRATING: look at the road, eyes open", (15, 35), YELLOW, 0.7, 2)
    cv2.rectangle(frame, (15, 50), (w - 15, 66), WHITE, 1)
    cv2.rectangle(frame, (15, 50), (15 + int((w - 30) * progress), 66), YELLOW, -1)
