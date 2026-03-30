import threading

import numpy as np
import sounddevice as sd


class AudioBuffer:
    """Continuous microphone capture with a small ring buffer.

    The buffer stores a fixed number of samples.  When more audio than the
    capacity is injected at once only the most recent portion is kept.
    """

    def __init__(
        self,
        samplerate: int = 16000,
        channels: int = 1,
        buffer_seconds: float = 1.0,
    ) -> None:
        self.samplerate = samplerate
        self.channels = channels
        self.capacity = int(buffer_seconds * samplerate)
        if self.capacity <= 0:
            raise ValueError("buffer_seconds too small")
        self.buffer = np.zeros((self.capacity, channels), dtype=np.float32)
        self.write_pos = 0
        self.lock = threading.Lock()
        self.stream = sd.InputStream(
            samplerate=samplerate,
            channels=channels,
            callback=self._callback,
        )
        self.stream.start()

    def _callback(self, indata: np.ndarray, frames: int, time, status) -> None:
        if status:
            print(f"[AudioBuffer] {status}")
        with self.lock:
            n = len(indata)
            pos = self.write_pos
            end = pos + n
            if end <= self.capacity:
                self.buffer[pos:end] = indata
            else:
                part1 = self.capacity - pos
                self.buffer[pos:] = indata[:part1]
                self.buffer[: end - self.capacity] = indata[part1:]
            self.write_pos = end % self.capacity

    def read(self, duration: float) -> np.ndarray:
        """Return the latest ``duration`` seconds of audio."""
        frames = int(duration * self.samplerate)
        if frames <= 0:
            return np.zeros(0, dtype=np.float32)
        frames = min(frames, self.capacity)
        with self.lock:
            pos = (self.write_pos - frames) % self.capacity
            if pos + frames <= self.capacity:
                data = self.buffer[pos : pos + frames]
            else:
                part1 = self.capacity - pos
                data = np.concatenate(
                    [self.buffer[pos:], self.buffer[: frames - part1]], axis=0
                )
        return data.squeeze().copy()

    def inject(self, audio: np.ndarray) -> None:
        """Insert ``audio`` into the buffer as if it were recorded."""
        if audio.ndim == 1:
            audio = audio.reshape(-1, self.channels)
        n = len(audio)
        if n == 0:
            return
        # Only keep the most recent ``capacity`` samples to avoid overflow
        if n > self.capacity:
            audio = audio[-self.capacity :]
            n = self.capacity
        with self.lock:
            pos = self.write_pos
            end = pos + n
            if end <= self.capacity:
                self.buffer[pos:end] = audio
            else:
                part1 = self.capacity - pos
                self.buffer[pos:] = audio[:part1]
                self.buffer[: end - self.capacity] = audio[part1:]
            self.write_pos = end % self.capacity

    def close(self) -> None:
        self.stream.stop()
        self.stream.close()
