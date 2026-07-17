from __future__ import annotations

import logging
import os
import threading
import time
from typing import Any, Literal, Optional

import cv2
import numpy as np

logger = logging.getLogger("wallcrossing.rtsp")

_GST_AVAILABLE: Optional[bool] = None
_GST_LOAD_ATTEMPTED = False
_GST: Any | None = None
_GST_IMPORT_ERROR: Exception | None = None
_GST_LOAD_LOCK = threading.Lock()
_FFMPEG_OPEN_LOCK = threading.Lock()
_DECODER_WARMUP_FRAMES = 5


class FrameConversionError(ValueError):
    """Raw GStreamer frame layout is incompatible with its negotiated caps."""


_DEPAY = {
    "h264": "rtph264depay ! h264parse",
    "h265": "rtph265depay ! h265parse",
}

GStreamerFormat = Literal["NV12", "I420", "BGR"]
_GSTREAMER_FORMATS: tuple[GStreamerFormat, ...] = ("NV12", "I420", "BGR")


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


def _load_gst() -> Any | None:
    """Load Python GStreamer bindings once. Returns Gst or None."""
    global _GST, _GST_IMPORT_ERROR, _GST_LOAD_ATTEMPTED
    with _GST_LOAD_LOCK:
        if _GST_LOAD_ATTEMPTED:
            return _GST
        try:
            import gi

            gi.require_version("Gst", "1.0")
            from gi.repository import Gst

            Gst.init(None)
            _GST = Gst
        except Exception as exc:  # pragma: no cover - depends on host packages
            _GST_IMPORT_ERROR = exc
            _GST = None
        finally:
            _GST_LOAD_ATTEMPTED = True
        return _GST


def _raw_video_to_bgr(data: np.ndarray, width: int, height: int, fmt: str) -> np.ndarray:
    return _raw_video_to_bgr_strided(data, width, height, fmt, None, None)


def _plane(data: np.ndarray, offset: int, rows: int, stride: int, used: int) -> np.ndarray:
    if offset < 0 or rows <= 0 or stride < used or used <= 0:
        raise FrameConversionError(
            f"invalid plane layout offset={offset} rows={rows} stride={stride} used={used}"
        )
    needed = offset + rows * stride
    if data.size < needed:
        raise FrameConversionError(f"buffer too small for plane: {data.size} < {needed}")
    return data[offset:needed].reshape((rows, stride))[:, :used]


def _raw_video_to_bgr_strided(
    data: np.ndarray,
    width: int,
    height: int,
    fmt: str,
    strides: list[int] | None,
    offsets: list[int] | None,
) -> np.ndarray:
    fmt = str(fmt)
    if width <= 0 or height <= 0:
        raise FrameConversionError(f"invalid frame size {width}x{height}")
    if fmt in {"NV12", "I420"} and (width % 2 or height % 2):
        raise FrameConversionError(f"{fmt} requires even dimensions, got {width}x{height}")

    if strides is None or offsets is None:
        if fmt == "NV12":
            strides = [width, width]
            offsets = [0, width * height]
        elif fmt == "I420":
            chroma_size = (width // 2) * (height // 2)
            strides = [width, width // 2, width // 2]
            offsets = [0, width * height, width * height + chroma_size]
        elif fmt in {"BGR", "RGB"}:
            strides = [width * 3]
            offsets = [0]
        elif fmt in {"BGRA", "BGRx", "RGBA", "RGBx"}:
            strides = [width * 4]
            offsets = [0]
        else:
            raise FrameConversionError(f"unsupported GStreamer raw format {fmt}")

    required_planes = 2 if fmt == "NV12" else 3 if fmt == "I420" else 1
    if len(strides) < required_planes or len(offsets) < required_planes:
        raise FrameConversionError(
            f"{fmt} needs {required_planes} planes, got strides={len(strides)} offsets={len(offsets)}"
        )

    if fmt == "NV12":
        y = _plane(data, offsets[0], height, strides[0], width)
        uv = _plane(data, offsets[1], height // 2, strides[1], width)
        packed = np.concatenate((y.reshape(-1), uv.reshape(-1)))
        raw = packed.reshape((height * 3 // 2, width))
        return cv2.cvtColor(raw, cv2.COLOR_YUV2BGR_NV12)
    if fmt == "I420":
        y = _plane(data, offsets[0], height, strides[0], width)
        u = _plane(data, offsets[1], height // 2, strides[1], width // 2)
        v = _plane(data, offsets[2], height // 2, strides[2], width // 2)
        packed = np.concatenate((y.reshape(-1), u.reshape(-1), v.reshape(-1)))
        raw = packed.reshape((height * 3 // 2, width))
        return cv2.cvtColor(raw, cv2.COLOR_YUV2BGR_I420)
    if fmt == "BGR":
        row = _plane(data, offsets[0], height, strides[0], width * 3)
        return row.reshape((height, width, 3)).copy()
    if fmt == "RGB":
        row = _plane(data, offsets[0], height, strides[0], width * 3)
        rgb = row.reshape((height, width, 3))
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    if fmt in {"BGRA", "BGRx"}:
        row = _plane(data, offsets[0], height, strides[0], width * 4)
        return row.reshape((height, width, 4))[:, :, :3].copy()
    if fmt in {"RGBA", "RGBx"}:
        row = _plane(data, offsets[0], height, strides[0], width * 4)
        rgba = row.reshape((height, width, 4))
        return cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)
    raise FrameConversionError(f"unsupported GStreamer raw format {fmt}")


def convert_gstreamer_frame(frame: np.ndarray, sink_format: GStreamerFormat = "BGR") -> np.ndarray:
    """Compatibility helper for tools/tests; live capture uses the same converter."""
    if frame.dtype != np.uint8:
        raise FrameConversionError(f"frame dtype must be uint8, got {frame.dtype}")
    if frame.ndim == 2 and sink_format in {"NV12", "I420"}:
        packed_rows, width = frame.shape
        if packed_rows * 2 % 3:
            raise FrameConversionError(f"invalid packed {sink_format} shape {frame.shape}")
        height = packed_rows * 2 // 3
        return _raw_video_to_bgr_strided(
            frame.reshape(-1), width, height, sink_format, None, None
        )
    if frame.ndim == 2:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    if frame.ndim != 3 or frame.shape[2] < 3:
        raise FrameConversionError(f"frame has unsupported shape {frame.shape}")
    return np.ascontiguousarray(frame[:, :, :3])


def _opencv_gstreamer_pipeline(rtsp_url: str, codec: str, transport: str) -> str:
    """OpenCV-GStreamer fallback converting decoder output to BGR."""
    return (
        f"rtspsrc location={rtsp_url} latency=500 protocols={transport} ! "
        f"{_DEPAY[codec]} ! mppvideodec ! "
        "queue leaky=downstream max-size-buffers=1 ! "
        "videoconvert ! video/x-raw,format=BGR ! "
        "appsink drop=true max-buffers=1 sync=false"
    )


def _gstreamer_pipeline(
    rtsp_url: str,
    codec: str,
    sink_format: GStreamerFormat = "NV12",
    transport: str = "tcp",
) -> str:
    """Hardware-decoded (rkmpp) pipeline for RK3588.

    Python-GStreamer reads raw NV12/I420 buffers directly and converts them to
    BGR in OpenCV according to negotiated caps. OpenCV-GStreamer is the fallback
    when Python bindings or direct raw formats are unavailable.

    NOTE: no videorate here — with live RTSP buffers (no duration) its max-rate
    path hits a glib assertion and abort()s the whole process.
    """
    if transport not in {"tcp", "udp"}:
        raise ValueError(f"unsupported RTSP transport {transport}")
    prefix = f"rtspsrc location={rtsp_url} latency=500 protocols={transport} ! {_DEPAY[codec]} ! "
    appsink = "appsink name=sink drop=true max-buffers=1 sync=false"
    if sink_format == "NV12":
        return prefix + "mppvideodec format=NV12 ! video/x-raw,format=NV12 ! " + appsink
    if sink_format == "I420":
        return prefix + "mppvideodec format=I420 ! video/x-raw,format=I420 ! " + appsink
    if sink_format == "BGR":
        return prefix + "mppvideodec format=BGR ! video/x-raw,format=BGR ! " + appsink
    raise ValueError(f"unsupported GStreamer sink format {sink_format}")


class _GstAppSinkCapture:
    """Small cv2.VideoCapture-like wrapper around GStreamer appsink."""

    is_direct_gst = True

    def __init__(self, pipeline: str, camera_id: str):
        self.camera_id = camera_id
        self._opened = False
        self._pipeline = None
        self._sink = None
        self._bus = None
        self._last_error: str | None = None
        self._conversion_error = False
        self._gst = _load_gst()
        if self._gst is None:
            self._last_error = f"python Gst unavailable: {_GST_IMPORT_ERROR!r}"
            return
        try:
            self._pipeline = self._gst.parse_launch(pipeline)
            self._sink = self._pipeline.get_by_name("sink")
            self._bus = self._pipeline.get_bus()
            if self._sink is None:
                self._last_error = "appsink named 'sink' not found"
                self.release()
                return
            ret = self._pipeline.set_state(self._gst.State.PLAYING)
            if ret == self._gst.StateChangeReturn.FAILURE:
                self._last_error = "failed to set pipeline PLAYING"
                self.release()
                return
            self._opened = True
        except Exception as exc:
            self._last_error = str(exc)
            self.release()

    def isOpened(self) -> bool:
        return self._opened

    @property
    def last_error(self) -> str | None:
        return self._last_error

    @property
    def format_incompatible(self) -> bool:
        if self._conversion_error:
            return True
        error = (self._last_error or "").lower()
        return any(
            marker in error
            for marker in (
                "not-negotiated",
                "could not link",
                "unsupported gstreamer raw format",
                "invalid plane layout",
                "needs 2 planes",
                "needs 3 planes",
                "requires even dimensions",
                "buffer too small for plane",
            )
        )

    def _drain_bus(self) -> None:
        if self._bus is None or self._gst is None:
            return
        mask = self._gst.MessageType.ERROR | self._gst.MessageType.EOS
        while True:
            msg = self._bus.pop_filtered(mask)
            if msg is None:
                return
            if msg.type == self._gst.MessageType.ERROR:
                err, debug = msg.parse_error()
                self._last_error = f"{err}; {debug}"
                self._opened = False
                return
            if msg.type == self._gst.MessageType.EOS:
                self._last_error = "EOS"
                self._opened = False
                return

    def read(self) -> tuple[bool, np.ndarray | None]:
        if not self._opened or self._sink is None or self._gst is None:
            return False, None
        self._drain_bus()
        if not self._opened:
            return False, None
        sample = self._sink.emit("try-pull-sample", int(0.2 * self._gst.SECOND))
        if sample is None:
            self._drain_bus()
            return False, None
        try:
            return True, self._sample_to_bgr(sample)
        except FrameConversionError as exc:
            self._last_error = str(exc)
            self._conversion_error = True
            return False, None
        except Exception as exc:
            self._last_error = str(exc)
            return False, None

    def _sample_to_bgr(self, sample) -> np.ndarray:
        caps = sample.get_caps()
        if caps is None or caps.get_size() == 0:
            raise ValueError("sample has no caps")
        struct = caps.get_structure(0)
        fmt = struct.get_value("format")
        width = int(struct.get_value("width"))
        height = int(struct.get_value("height"))
        buf = sample.get_buffer()
        strides = offsets = None
        try:
            import gi

            gi.require_version("GstVideo", "1.0")
            from gi.repository import GstVideo

            meta = GstVideo.buffer_get_video_meta(buf)
            if meta is not None:
                strides = [int(meta.stride[i]) for i in range(int(meta.n_planes))]
                offsets = [int(meta.offset[i]) for i in range(int(meta.n_planes))]
        except Exception:
            strides = offsets = None

        ok, map_info = buf.map(self._gst.MapFlags.READ)
        if not ok:
            raise ValueError("could not map sample buffer")
        try:
            data = np.frombuffer(map_info.data, dtype=np.uint8)
            return _raw_video_to_bgr_strided(data, width, height, fmt, strides, offsets)
        finally:
            buf.unmap(map_info)

    def release(self) -> None:
        if self._pipeline is not None and self._gst is not None:
            self._pipeline.set_state(self._gst.State.NULL)
        self._opened = False
        self._pipeline = None
        self._sink = None
        self._bus = None


class RtspReader:
    """Reads an RTSP stream in a background thread, keeping only the latest frame.

    Old frames are dropped so consumers always see fresh data (low latency). On
    read failure it reconnects with capped backoff.

    target_fps caps how many frames per second are converted to BGR and stored:
    with the gstreamer backend appsink drops stale frames and Python converts
    only the kept raw frame; with the opencv/FFmpeg backend the thread grab()s
    every frame but only retrieve()s frames it will keep.
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
        self._gst_format_index = 0
        self._active_gst_format: str = _GSTREAMER_FORMATS[0]
        self._use_opencv_gst_fallback = False
        self.connected = False
        self.reconnect_count = 0
        self.last_frame_mono: float = 0.0

    def _advance_gstreamer_format(self) -> bool:
        if self._use_opencv_gst_fallback:
            return False
        if self._gst_format_index >= len(_GSTREAMER_FORMATS) - 1:
            self._use_opencv_gst_fallback = True
            self._active_gst_format = "OpenCV-BGR"
            logger.warning(
                "cam=%s direct gstreamer formats failed, falling back to OpenCV-GStreamer BGR",
                self.camera_id,
            )
            return True
        old = _GSTREAMER_FORMATS[self._gst_format_index]
        self._gst_format_index += 1
        new = _GSTREAMER_FORMATS[self._gst_format_index]
        logger.warning("cam=%s disabling gstreamer format=%s, trying %s", self.camera_id, old, new)
        return True

    def _open_gstreamer(self):
        gst = _load_gst()
        if gst is not None and not self._use_opencv_gst_fallback:
            fmt = _GSTREAMER_FORMATS[self._gst_format_index]
            cap = _GstAppSinkCapture(
                _gstreamer_pipeline(self.rtsp_url, self.codec, fmt, self.transport),
                self.camera_id,
            )
            if cap.isOpened():
                self._active_gst_format = fmt
                return cap
            if cap.last_error:
                logger.warning(
                    "cam=%s python-gst format=%s open failed: %s",
                    self.camera_id,
                    fmt,
                    cap.last_error,
                )
            if cap.format_incompatible and self._advance_gstreamer_format():
                return self._open_gstreamer()
            return cap
        elif gst is None and not gstreamer_available():
            logger.error(
                "cam=%s neither Python GStreamer nor OpenCV GStreamer is available: %r",
                self.camera_id,
                _GST_IMPORT_ERROR,
            )
            return cv2.VideoCapture()

        if not gstreamer_available():
            return cv2.VideoCapture()
        self._active_gst_format = "OpenCV-BGR"
        return cv2.VideoCapture(
            _opencv_gstreamer_pipeline(self.rtsp_url, self.codec, self.transport),
            cv2.CAP_GSTREAMER,
        )

    def _open(self):
        backend = self.decode_backend
        self._active_backend = backend

        if backend == "gstreamer":
            return self._open_gstreamer()

        # This env var is global; serialize option update with capture creation.
        opts = f"rtsp_transport;{self.transport}"
        if self.ffmpeg_video_codec:
            opts += f"|video_codec;{self.ffmpeg_video_codec}"
        with _FFMPEG_OPEN_LOCK:
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = opts
            cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name=f"rtsp-{self.camera_id}", daemon=True)
        self._thread.start()

    def _run(self) -> None:
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
            if self._active_backend == "gstreamer":
                logger.info("cam=%s connected (%s/%s)", self.camera_id, self._active_backend, self._active_gst_format)
            else:
                logger.info("cam=%s connected (%s)", self.camera_id, self._active_backend)

            interval = 1.0 / self.target_fps if self.target_fps > 0 else 0.0
            is_gst = self._active_backend == "gstreamer"
            is_direct_gst = bool(getattr(cap, "is_direct_gst", False))
            next_slot = 0.0

            failure_started: float | None = None
            warmup_remaining = _DECODER_WARMUP_FRAMES
            switch_gst_format = False
            while not self._stop.is_set():
                if is_gst:
                    if interval:
                        wait = next_slot - time.monotonic()
                        if wait > 0 and self._stop.wait(wait):
                            break
                    ok, frame = cap.read()
                    if ok and frame is not None:
                        failure_started = None
                        next_slot = time.monotonic() + interval
                    else:
                        now = time.monotonic()
                        failure_started = failure_started or now
                        failure_window = now - failure_started
                        max_failure_window = 2.0 if is_direct_gst else 10.0
                        incompatible = bool(
                            is_direct_gst and getattr(cap, "format_incompatible", False)
                        )
                        if incompatible or failure_window >= max_failure_window:
                            last_error = getattr(cap, "last_error", None)
                            if last_error:
                                logger.warning("cam=%s read failed: %s", self.camera_id, last_error)
                            logger.warning(
                                "cam=%s read failed for %.1fs, reconnecting",
                                self.camera_id,
                                failure_window,
                            )
                            if incompatible:
                                switch_gst_format = self._advance_gstreamer_format()
                            break
                        self._stop.wait(min(interval or 0.02, 0.2))
                        continue
                else:
                    if not cap.grab():
                        now = time.monotonic()
                        failure_started = failure_started or now
                        failure_window = now - failure_started
                        if failure_window >= 10.0:
                            logger.warning(
                                "cam=%s grab failed for %.1fs, reconnecting",
                                self.camera_id,
                                failure_window,
                            )
                            break
                        self._stop.wait(min(interval or 0.02, 0.2))
                        continue

                    if interval:
                        now = time.monotonic()
                        if now < next_slot:
                            continue

                    ok, frame = cap.retrieve()
                    if not ok or frame is None:
                        now = time.monotonic()
                        failure_started = failure_started or now
                        failure_window = now - failure_started
                        if failure_window >= 10.0:
                            logger.warning(
                                "cam=%s retrieve failed for %.1fs, reconnecting",
                                self.camera_id,
                                failure_window,
                            )
                            break
                        self._stop.wait(min(interval or 0.02, 0.2))
                        continue
                    failure_started = None
                    next_slot = time.monotonic() + interval

                if warmup_remaining > 0:
                    warmup_remaining -= 1
                    continue

                with self._lock:
                    self._latest = frame
                    self._frame_index += 1
                self.last_frame_mono = time.monotonic()

            cap.release()
            self.connected = False
            if not self._stop.is_set():
                self.reconnect_count += 1
                if switch_gst_format:
                    delay = self.reconnect_delay
                    continue
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
        with self._lock:
            self._latest = None

    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def join(self, timeout: float | None = 5.0) -> bool:
        if self._thread is None:
            return True
        self._thread.join(timeout=timeout)
        return not self._thread.is_alive()
