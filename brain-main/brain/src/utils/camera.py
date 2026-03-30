import cv2
import threading
from typing import Optional


class Camera:
    """Background frame grabber that always keeps the latest frame."""

    def __init__(self, camera_id: int = 0):
        self.cap = cv2.VideoCapture(camera_id)
        self.lock = threading.Lock()
        self.frame = None
        self.running = True
        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()

    def _reader(self) -> None:
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                continue
            with self.lock:
                self.frame = frame

    def read(self) -> Optional[cv2.typing.MatLike]:
        with self.lock:
            if self.frame is None:
                return None
            return self.frame.copy()

    def release(self) -> None:
        self.running = False
        self.thread.join()
        self.cap.release()
