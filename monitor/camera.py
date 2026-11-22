"""Opens the webcam (or a video file) and hands out frames with timestamps."""
import sys
import time

import cv2


class FrameSource:
    """Reads frames from a webcam or a video file.

    Each read returns (frame, t) where t is seconds since the start. For a webcam
    t is the real clock; for a video file it is the video's own time, so a
    recorded clip always gives the same results.
    """

    def __init__(self, camera_index=0, width=640, height=480, mirror=True, video_path=None):
        self.mirror = mirror
        self.is_file = video_path is not None
        if self.is_file:
            self.cap = cv2.VideoCapture(video_path)
            self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        else:
            # DirectShow opens webcams much faster on Windows.
            backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY
            self.cap = cv2.VideoCapture(camera_index, backend)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self.fps = None
        if not self.cap.isOpened():
            src = video_path if self.is_file else f"camera {camera_index}"
            raise RuntimeError(f"Could not open {src}. Check the camera index or file path.")
        self.frame_no = 0
        self.t0 = time.monotonic()

    def read(self):
        ok, frame = self.cap.read()
        if not ok:
            return None, None
        if self.mirror:
            frame = cv2.flip(frame, 1)
        t = self.frame_no / self.fps if self.is_file else time.monotonic() - self.t0
        self.frame_no += 1
        return frame, t

    def release(self):
        self.cap.release()
