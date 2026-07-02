from __future__ import annotations

import logging
import os
import threading
import time
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger("wallcrossing.rtsp")

_GST_AVAILABLE: Optional[bool] = None

_DEPAY = {
    "h264": "rtph264depay ! h264parse",
    "h265": "rtph265depay ! h265parse",
}


def gstreamer_available() -> bool:
    """True if this OpenCV build has GStreamer support (checked once, cached)."""
    global _GST_AVAILABLE
    if _GST_AVAILABLE is None:
        _GST_AVAILABLE = False
        for line in cv2.getBuildInformation().splitlines():
            if "GStreamer" in line:
                _GST_AVAILABLE = "YES" in line
                break
    return _GST_AVAILABLE


def _gstreamer_pipeline(rtsp_url: str, codec: str, max_fps: float) -> str:
    """Hardware-decoded (rkmpp) pipeline for RK3588. Requires OpenCV built with GStreamer.

    mppvideodec decodes on the VPU (cheap), but videoconvert (NV12 -> BGR) runs
    on the CPU at full resolution. videorate drop-only sits between them so only
    ~max_fps frames per second reach videoconvert — dropped frames cost nothing.
    """
    rate = ""
    if max_fps > 0:
        rate = f"videorate drop-only=true max-rate={max(1, int(round(max_fps)))} ! "
    return (
        f"rtspsrc location={rtsp_url} latency=200 protocols=tcp tcp-timeout=5000000 ! "
        f"{_DEPAY[codec]} ! mppvideodec ! "
        f"{rate}"
        "videoconvert ! video/x-raw,format=BGR ! "
        "appsink drop=true max-buffers=1 sync=false"
    )


class RtspReader:
    """Reads an RTSP stream in a background thread, keeping only the latest frame.

    Old frames are dropped so consumers always see fresh data (low latency). On
    read failure it reconnects with capped backoff.

    target_fps caps how many frames per second are converted to BGR and stored:
    with the gstreamer backend excess frames are dropped inside the pipeline
    (before videoconvert); with the opencv/FFmpeg backend the thread grab()s
    every frame (H.264/H.265 must still be decoded because frames reference
    each other) but only retrieve()s — i.e. pays the YUV->BGR conversion for —
    frames it will actually keep.
    """

    def __init__(
        self,
        camera_id: str,
        rtsp_url: str,
        decode_backend: str = "gstreamer",
        target_fps: float = 0.0,
        codec: str = "h264",
        ffmpeg_video_codec: str = "",
        reconnect_delay: float = 2.0,
        max_reconnect_delay: float = 30.0,
    ):
        self.camera_id = camera_id
        self.rtsp_url = rtsp_url
        self.decode_backend = decode_backend
        self.target_fps = target_fps
        self.codec = codec
        self.ffmpeg_video_codec = ffmpeg_video_codec
        self.reconnect_delay = reconnect_delay
        self.max_reconnect_delay = max_reconnect_delay

        self._lock = threading.Lock()
        self._latest: Optional[np.ndarray] = None
        self._frame_index = 0
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._active_backend = decode_backend
        self.connected = False
        self.reconnect_count = 0
        self.last_frame_mono: float = 0.0

    def _open(self) -> cv2.VideoCapture:
        backend = self.decode_backend
        if backend == "gstreamer" and not gstreamer_available():
            logger.error(
                "cam=%s OpenCV was built without GStreamer: falling back to FFmpeg "
                "SOFTWARE decode. This is very CPU-heavy on RK3588 — install an "
                "OpenCV build with GStreamer + rockchip-mpp to use the VPU.",
                self.camera_id,
            )
            backend = "opencv"
        self._active_backend = backend

        if backend == "gstreamer":
            return cv2.VideoCapture(
                _gstreamer_pipeline(self.rtsp_url, self.codec, self.target_fps),
                cv2.CAP_GSTREAMER,
            )

        # This env var is global for the whole process; all cameras share it.
        opts = "rtsp_transport;tcp"
        if self.ffmpeg_video_codec:
            opts += f"|video_codec;{self.ffmpeg_video_codec}"
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = opts
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
            if self._stop.is_set():
                cap.release()
                break
            if not cap.isOpened():
                self.connected = False
                logger.warning("cam=%s open failed, retry in %.1fs", self.camera_id, delay)
                self._stop.wait(delay)
                delay = min(delay * 2, self.max_reconnect_delay)
                self.reconnect_count += 1
                continue

            self.connected = True
            delay = self.reconnect_delay
            logger.info("cam=%s connected (%s)", self.camera_id, self._active_backend)

            # gstreamer already throttles in-pipeline (videorate); only the
            # FFmpeg path needs retrieve()-throttling here.
            throttle = (
                1.0 / self.target_fps
                if self.target_fps > 0 and self._active_backend != "gstreamer"
                else 0.0
            )
            next_retrieve = 0.0

            consecutive_failures = 0
            while not self._stop.is_set():
                if not cap.grab():
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

                if throttle:
                    now = time.monotonic()
                    if now < next_retrieve:
                        continue
                    next_retrieve = now + throttle

                ok, frame = cap.retrieve()
                if not ok or frame is None:
                    continue
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
        """Return (frame, index). The frame is shared read-only — do NOT draw on
        it in place; copy first. The reader never mutates a stored frame (each
        retrieve() allocates a fresh buffer), so holding a reference is safe."""
        with self._lock:
            return self._latest, self._frame_index

    def stop(self) -> None:
        self.request_stop()
        self.join()

    def request_stop(self) -> None:
        self._stop.set()

    def join(self) -> None:
        if self._thread is not None:
            self._thread.join(timeout=5.0)
