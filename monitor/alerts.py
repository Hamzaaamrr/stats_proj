"""On-screen warning banner and an optional beep."""
import sys
import threading
import time

import cv2

MESSAGES = {
    "Drowsy": "WAKE UP! Signs of drowsiness",
    "Distracted": "EYES ON THE ROAD!",
    "Using Phone": "PUT THE PHONE DOWN!",
    "Eating": "Eating while driving",
    "Drinking": "Drinking while driving",
}


class AlertManager:
    def __init__(self, audio=True, cooldown=4.0):
        self.audio = audio
        self.cooldown = cooldown
        self._last_beep = -1e9

    def trigger(self, behavior):
        """Plays a beep for a risky behavior, at most once per cooldown."""
        if not self.audio or behavior not in MESSAGES:
            return
        now = time.monotonic()
        if now - self._last_beep < self.cooldown:
            return
        self._last_beep = now
        urgent = behavior in ("Drowsy", "Using Phone", "Distracted")
        threading.Thread(target=_beep, args=(urgent,), daemon=True).start()

    @staticmethod
    def draw_banner(frame, behavior, confidence):
        if behavior not in MESSAGES:
            return
        h, w = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 60), (0, 0, 220), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), 6)
        text = f"{MESSAGES[behavior]}  ({confidence:.0%})"
        cv2.putText(frame, text, (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2,
                    cv2.LINE_AA)


def _beep(urgent):
    if sys.platform.startswith("win"):
        import winsound
        if urgent:
            for _ in range(3):
                winsound.Beep(1500, 200)
        else:
            winsound.Beep(1000, 300)
    else:
        print("\a", end="", flush=True)  # terminal bell on macOS/Linux
