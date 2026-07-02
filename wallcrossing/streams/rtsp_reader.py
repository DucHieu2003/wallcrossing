from __future__ import annotations

import logging
import os
import threading
import time
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger("wallcrossing.rtsp")


def _gstreamer_pipeline(rtsp_url: str) -> str:
    """Hardware-decoded (rkmpp) pipeline for RK3588. Requires OpenCV built with GStreamer."""
    return (
        f"rtspsrc location={rtsp_url} latency=200 protocols=tcp ! "
        "rtph264depay ! h264parse ! mppvideodec ! "
        "videoconvert ! video/x-raw,format=BGR ! "
        "appsink drop=true max-buffers=1 sync=false"
    )


class RtspReader:
    """Reads an RTSP stream in a background thread, keeping only the latest frame.

    Old frames are dropped so consumers always see fresh data (low latency). On
    read failure it reconnects with capped backoff.
    """

    def __init__(
        self,
        camera_id: str,
        rtsp_url: str,
        decode_backend: str = "gstreamer",
        reconnect_delay: float = 2.0,
        max_reconnect_delay: float = 30.0,
    ):
        self.camera_id = camera_id
        self.rtsp_url = rtsp_url
        self.decode_backend = decode_backend
        self.reconnect_delay = reconnect_delay
        self.max_reconnect_delay = max_reconnect_delay

        self._lock = threading.Lock()
        self._latest: Optional[np.ndarray] = None
        self._frame_index = 0
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.connected = False
        self.reconnect_count = 0
        self.last_frame_mono: float = 0.0

    def _open(self) -> cv2.VideoCapture:
        if self.decode_backend == "gstreamer":
            return cv2.VideoCapture(_gstreamer_pipeline(self.rtsp_url), cv2.CAP_GSTREAMER)
        os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
        cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name=f"rtsp-{self.camera_id}", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        delay = self.reconnect_delay
        while not self._stop.is_set():
            cap = self._open()
            if not cap.isOpened():
                self.connected = False
                logger.warning("cam=%s open failed, retry in %.1fs", self.camera_id, delay)
                self._stop.wait(delay)
                delay = min(delay * 2, self.max_reconnect_delay)
                self.reconnect_count += 1
                continue

            self.connected = True
            delay = self.reconnect_delay
            logger.info("cam=%s connected", self.camera_id)

            consecutive_failures = 0
            while not self._stop.is_set():
                ok, frame = cap.read()
                if not ok or frame is None:
                    consecutive_failures += 1
                    if consecutive_failures >= 30:
                        logger.warning(
                            "cam=%s read failed %d times, reconnecting",
                            self.camera_id,
                            consecutive_failures,
                        )
                        break
                    time.sleep(0.02)
                    continue
                consecutive_failures = 0
                with self._lock:
                    self._latest = frame
                    self._frame_index += 1
                self.last_frame_mono = time.monotonic()

            cap.release()
            self.connected = False
            if not self._stop.is_set():
                self.reconnect_count += 1
                self._stop.wait(delay)

    def read_latest(self) -> tuple[Optional[np.ndarray], int]:
        with self._lock:
            if self._latest is None:
                return None, self._frame_index
            return self._latest.copy(), self._frame_index

    def stop(self) -> None:
        self.request_stop()
        self.join()

    def request_stop(self) -> None:
        self._stop.set()

    def join(self) -> None:
        if self._thread is not None:
            self._thread.join(timeout=5.0)
