"""
zmq_image_publisher.py
======================
Standalone ZMQ image-streaming utilities for icub-lerobot.
"""

import logging
import time
from typing import Optional

import cv2
import numpy as np

try:
    import zmq
except ImportError:
    zmq = None

logger = logging.getLogger(__name__)

_GLOBAL_ZMQ_CONTEXT = None


def _get_global_context():
    """Return (and lazily create) the module-level ZMQ context."""
    global _GLOBAL_ZMQ_CONTEXT
    if zmq is None:
        raise ImportError("pyzmq is required for VR image streaming (`pixi install`).")
    if _GLOBAL_ZMQ_CONTEXT is None:
        _GLOBAL_ZMQ_CONTEXT = zmq.Context()
    return _GLOBAL_ZMQ_CONTEXT


class BasePublisher:
    """Base class for all ZMQ publishers."""

    def __init__(
        self,
        host: str,
        port: int,
        socket_type: Optional[int] = None,
        context=None,
    ) -> None:
        if zmq is None:
            raise ImportError("pyzmq is required for VR image streaming (`pixi install`).")
        self._host = host
        self._port = port
        self._socket = None
        self._context = context or _get_global_context()
        self._init_socket(socket_type if socket_type is not None else zmq.PUB)

    def _init_socket(self, socket_type: int) -> None:
        self._socket = self._context.socket(socket_type)
        self._socket.setsockopt(zmq.SNDHWM, 1)
        self._socket.setsockopt(zmq.LINGER, 0)
        addr = f"tcp://*:{self._port}"

        attempts = 20
        for attempt in range(attempts):
            try:
                self._socket.bind(addr)
                return
            except zmq.ZMQError as exc:
                if attempt == attempts - 1:
                    logger.error(f"Failed to initialise socket on port {self._port}: {exc}")
                    raise
                time.sleep(0.1)

    def stop(self) -> None:
        """Close the publisher socket."""
        if self._socket:
            self._socket.close()
            self._socket = None


class ZMQCompressedImageTransmitter(BasePublisher):
    """ZMQ PUB publisher that sends JPEG-compressed images as multipart messages."""

    def __init__(self, host: str, port: int, quality: int = 90) -> None:
        super().__init__(host, port)
        self._quality = quality

    def send_image(self, rgb_image: np.ndarray) -> None:
        _, buffer = cv2.imencode(
            ".jpg", rgb_image, [int(cv2.IMWRITE_JPEG_QUALITY), self._quality]
        )
        assert self._socket is not None, "Socket is not initialised"
        self._socket.send_multipart(
            [b"image", np.array(buffer).tobytes()], zmq.NOBLOCK
        )

