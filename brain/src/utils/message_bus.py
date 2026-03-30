import threading
import time
from typing import Callable, Dict

import msgpack
import msgpack_numpy
import numpy as np

import zmq


class MessageBus:
    """Simple PUB/SUB message bus using ZeroMQ."""

    def __init__(self, address: str = "tcp://127.0.0.1:5555") -> None:
        self.ctx = zmq.Context.instance()
        self.address = address
        self.pub = self.ctx.socket(zmq.PUB)
        self.pub.bind(address)
        self._subs: Dict[str, zmq.Socket] = {}
        self._threads: list[threading.Thread] = []
        self._count: Dict[str, int] = {}
        self._start_time = time.time()

    def publish(self, topic: str, data: bytes) -> None:
        """Broadcast ``data`` under ``topic``."""
        self.pub.send_multipart([topic.encode(), data])
        self._count[topic] = self._count.get(topic, 0) + 1

    def publish_array(self, topic: str, array: np.ndarray) -> None:
        """Serialize and publish ``array`` using msgpack."""
        packed = msgpack.dumps(array, default=msgpack_numpy.encode)
        self.publish(topic, packed)

    def subscribe(self, topic: str, handler: Callable[[bytes], None]) -> None:
        """Listen on ``topic`` and invoke ``handler`` for each message."""
        sub = self.ctx.socket(zmq.SUB)
        sub.connect(self.address)
        sub.setsockopt_string(zmq.SUBSCRIBE, topic)

        def loop() -> None:
            while True:
                _t, msg = sub.recv_multipart()
                handler(msg)

        t = threading.Thread(target=loop, daemon=True)
        t.start()
        self._subs[topic] = sub
        self._threads.append(t)

    def subscribe_array(self, topic: str, handler: Callable[[np.ndarray], None]) -> None:
        """Subscribe to a topic expecting msgpack arrays."""
        def wrap(msg: bytes) -> None:
            arr = msgpack.loads(msg, object_hook=msgpack_numpy.decode)
            handler(arr)

        self.subscribe(topic, wrap)

    def get_rates(self) -> Dict[str, float]:
        """Return messages per second for each topic since last call."""
        now = time.time()
        elapsed = max(now - self._start_time, 1e-6)
        rates = {t: c / elapsed for t, c in self._count.items()}
        self._count.clear()
        self._start_time = now
        return rates
