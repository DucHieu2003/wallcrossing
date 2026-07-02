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


# Whether OpenCV's GStreamer capture accepts NV12 appsink caps.
# None = not yet known; decided once by the first reader that tries, shared by all.
_NV12_SUPPORTED: Optional[bool] = None


def _gstreamer_pipeline(rtsp_url: str, codec: str, transport: str, fmt: str) -> str:
    """Hardware-decoded (rkmpp) pipeline for RK3588. Requires OpenCV built with GStreamer.

    Preferred shape (fmt="NV12"): the decoder hands NV12 straight to appsink —
    zero per-frame CPU in the pipeline. Excess frames are dropped by appsink as
    cheap NV12 buffers; the reader converts to BGR in Python only at target_fps.

    Fallback shape (fmt="BGR", when OpenCV can't take NV12 from appsink):
    videoconvert runs on the CPU at full stream rate; the leaky queue drops
    frames instead of blocking when it falls behind. At 2560x1440 this costs
    real CPU per camera — NV12 is much cheaper.

    Nothing here may ever block upstream: backpressure onto the decoder starves
    its DMA pool and stalls the TCP socket, killing rtspsrc within seconds.
    NOTE: no videorate — with live RTSP buffers (no duration) its max-rate path
    hits a glib assertion and abort()s the whole process.
    """
    src = (
        f"rtspsrc location={rtsp_url} latency=200 protocols={transport} ! "
        f"{_DEPAY[codec]} ! mppvideodec format=NV12 ! "
    )
    if fmt == "NV12":
        return src + "video/x-raw,format=NV12 ! appsink drop=true max-buffers=1 sync=false"
    return src + (
        "queue leaky=downstream max-size-buffers=1 ! "
        "videoconvert n-threads=2 ! video/x-raw,format=BGR ! "
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
        transport: str = "tcp",
        ffmpeg_video_codec: str = "",
        reconnect_delay: float = 2.0,
        max_reconnect_delay: float = 30.0,
        initial_delay: float = 0.0,
    ):
        self.camera_id = camera_id
        self.rtsp_url = rtsp_url
        self.decode_backend = decode_backend
        self.target_fps = target_fps
        self.codec = codec
        self.transport = transport
        self.ffmpeg_video_codec = ffmpeg_video_codec
        self.reconnect_delay = reconnect_delay
        self.max_reconnect_delay = max_reconnect_delay
        self.initial_delay = initial_delay

        self._lock = threading.Lock()
        self._latest: Optional[np.ndarray] = None
        self._frame_index = 0
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._active_backend = decode_backend
        self._gst_format = "BGR"
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
            global _NV12_SUPPORTED
            fmt = "BGR" if _NV12_SUPPORTED is False else "NV12"
            cap = cv2.VideoCapture(
                _gstreamer_pipeline(self.rtsp_url, self.codec, self.transport, fmt),
                cv2.CAP_GSTREAMER,
            )
            if fmt == "NV12" and not cap.isOpened() and _NV12_SUPPORTED is None:
                _NV12_SUPPORTED = False
                logger.info("OpenCV khong nhan appsink NV12, chuyen sang videoconvert BGR")
                cap.release()
                cap = cv2.VideoCapture(
                    _gstreamer_pipeline(self.rtsp_url, self.codec, self.transport, "BGR"),
                    cv2.CAP_GSTREAMER,
                )
            self._gst_format = fmt if _NV12_SUPPORTED is not False else "BGR"
            return cap

        # This env var is global for the whole process; all cameras share it.
        opts = f"rtsp_transport;{self.transport}"
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
        global _NV12_SUPPORTED
        # stagger startup so N cameras don't all get RTSP handshakes at once
        if self.initial_delay > 0 and self._stop.wait(self.initial_delay):
            return
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

            # Two throttle styles:
            # - gstreamer: sleep until the next slot, then pull. While we sleep
            #   the leaky queue drops frames as cheap NV12 before videoconvert.
            # - opencv/FFmpeg: grab() every frame (the decoder must consume all
            #   of them), but retrieve() (YUV->BGR) only once per slot.
            interval = 1.0 / self.target_fps if self.target_fps > 0 else 0.0
            is_gst = self._active_backend == "gstreamer"
            next_slot = 0.0

            consecutive_failures = 0
            session_frames = 0
            while not self._stop.is_set():
                if is_gst and interval:
                    wait = next_slot - time.monotonic()
                    if wait > 0 and self._stop.wait(wait):
                        break

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

                if not is_gst and interval:
                    now = time.monotonic()
                    if now < next_slot:
                        continue
                    next_slot = now + interval

                ok, frame = cap.retrieve()
                if not ok or frame is None:
                    continue
                if frame.ndim == 2:
                    # appsink delivered raw NV12 (h*3/2, w); convert here, at
                    # target_fps only — this is the whole point of the NV12 path
                    frame = cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_NV12)
                if is_gst:
                    next_slot = time.monotonic() + interval
                    if _NV12_SUPPORTED is None and self._gst_format == "NV12":
                        _NV12_SUPPORTED = True
                with self._lock:
                    self._latest = frame
                    self._frame_index += 1
                session_frames += 1
                self.last_frame_mono = time.monotonic()

            # NV12 caps negotiated but no frame ever arrived: the format is not
            # actually usable through OpenCV — fall back to BGR for everyone.
            if (
                is_gst
                and session_frames == 0
                and self._gst_format == "NV12"
                and _NV12_SUPPORTED is None
            ):
                _NV12_SUPPORTED = False
                logger.info("cam=%s NV12 khong ra frame, chuyen sang videoconvert BGR", self.camera_id)

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
