"""
zmq_image_publisher.py
======================
Standalone ZMQ image-streaming utilities for icub-lerobot.

Originally extracted from beavr-bot/src/beavr/teleop/common/network/publisher.py
(commit date: 2026-03-17).  This copy removes the dependency on the external
beavr-bot package so that icub-lerobot can operate without it.

Only the classes required by teleop_mujoco.py are included:
  - BasePublisher
  - ZMQCompressedImageTransmitter
"""

import logging
import time
from typing import Optional

import cv2
import numpy as np
import zmq

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global ZMQ context (one per process, mirroring beavr/teleop/common/network/utils.py)
# ---------------------------------------------------------------------------
_GLOBAL_ZMQ_CONTEXT: Optional[zmq.Context] = None


def _get_global_context() -> zmq.Context:
    """Return (and lazily create) the module-level ZMQ context."""
    global _GLOBAL_ZMQ_CONTEXT
    if _GLOBAL_ZMQ_CONTEXT is None:
        _GLOBAL_ZMQ_CONTEXT = zmq.Context()
    return _GLOBAL_ZMQ_CONTEXT


# ---------------------------------------------------------------------------
# BasePublisher
# ---------------------------------------------------------------------------

class BasePublisher:
    """Base class for all ZMQ publishers."""

    def __init__(
        self,
        host: str,
        port: int,
        socket_type: int = zmq.PUB,
        context: Optional[zmq.Context] = None,
    ) -> None:
        self._host = host
        self._port = port
        self._socket: Optional[zmq.Socket] = None
        self._context = context or _get_global_context()
        self._init_socket(socket_type)

    def _init_socket(self, socket_type: int) -> None:
        self._socket = self._context.socket(socket_type)
        self._socket.setsockopt(zmq.SNDHWM, 1)  # Only keep the latest message
        self._socket.setsockopt(zmq.LINGER, 0)  # Release the port promptly on close()
        addr = f"tcp://*:{self._port}"

        # A just-closed publisher from a previous session may not have released
        # the port yet (libzmq tears down the socket asynchronously), so retry
        # briefly instead of failing the whole stream on a transient EADDRINUSE.
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


# ---------------------------------------------------------------------------
# ZMQCompressedImageTransmitter
# ---------------------------------------------------------------------------

class ZMQCompressedImageTransmitter(BasePublisher):
    """ZMQ PUB publisher that sends JPEG-compressed images as multipart messages.

    Usage::

        pub = ZMQCompressedImageTransmitter(host="*", port=10505)
        pub.send_image(rgb_array)   # numpy (H, W, 3) uint8

    Compatible with beavr-bot's original ``ZMQCompressedImageTransmitter`` API.
    """

    def __init__(self, host: str, port: int, quality: int = 90) -> None:
        super().__init__(host, port, zmq.PUB)
        self._quality = quality

    def send_image(self, rgb_image: np.ndarray) -> None:
        """Compress *rgb_image* as JPEG and publish it via ZMQ.

        Args:
            rgb_image: RGB image array with shape (H, W, 3), dtype uint8.
        """
        _, buffer = cv2.imencode(
            ".jpg", rgb_image, [int(cv2.IMWRITE_JPEG_QUALITY), self._quality]
        )
        assert self._socket is not None, "Socket is not initialised"
        self._socket.send_multipart(
            [b"image", np.array(buffer).tobytes()], zmq.NOBLOCK
        )
