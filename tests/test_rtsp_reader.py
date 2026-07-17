import cv2
import numpy as np
import pytest

from wallcrossing.streams.rtsp_reader import (
    FrameConversionError,
    _GstAppSinkCapture,
    _gstreamer_pipeline,
    _raw_video_to_bgr_strided,
    convert_gstreamer_frame,
)


def _solid_bgr(color: tuple[int, int, int], height: int = 32, width: int = 32) -> np.ndarray:
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:, :] = color
    return frame


def _i420_from_bgr(frame: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame, cv2.COLOR_BGR2YUV_I420)


def _nv12_from_bgr(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    i420 = _i420_from_bgr(frame)
    y = i420[:h, :]
    u = i420[h : h + h // 4, :].reshape(-1)
    v = i420[h + h // 4 :, :].reshape(-1)
    uv = np.empty(u.size + v.size, dtype=np.uint8)
    uv[0::2] = u
    uv[1::2] = v
    return np.vstack([y, uv.reshape(h // 2, w)])


def _padded_buffer(planes: list[np.ndarray], strides: list[int], offsets: list[int]) -> np.ndarray:
    size = max(offset + plane.shape[0] * stride for plane, stride, offset in zip(planes, strides, offsets))
    data = np.zeros(size, dtype=np.uint8)
    for plane, stride, offset in zip(planes, strides, offsets):
        rows, used = plane.shape
        view = data[offset : offset + rows * stride].reshape(rows, stride)
        view[:, :used] = plane
    return data


def test_convert_gstreamer_frame_converts_nv12_to_bgr_color():
    original = _solid_bgr((20, 60, 220))
    converted = convert_gstreamer_frame(_nv12_from_bgr(original), "NV12")

    mean_b, mean_g, mean_r = converted.reshape(-1, 3).mean(axis=0)
    assert converted.shape == original.shape
    assert mean_r > mean_g > mean_b


def test_convert_gstreamer_frame_converts_i420_to_bgr_color():
    original = _solid_bgr((210, 70, 20))
    converted = convert_gstreamer_frame(_i420_from_bgr(original), "I420")

    mean_b, mean_g, mean_r = converted.reshape(-1, 3).mean(axis=0)
    assert converted.shape == original.shape
    assert mean_b > mean_g > mean_r


def test_i420_live_converter_handles_padded_planes_and_height_not_divisible_by_four():
    original = _solid_bgr((200, 80, 30), height=34, width=32)
    packed = _i420_from_bgr(original).reshape(-1)
    y_size = 32 * 34
    chroma_size = 16 * 17
    y = packed[:y_size].reshape(34, 32)
    u = packed[y_size : y_size + chroma_size].reshape(17, 16)
    v = packed[y_size + chroma_size :].reshape(17, 16)
    strides = [40, 24, 24]
    offsets = [0, 40 * 34 + 11, 40 * 34 + 11 + 24 * 17 + 13]
    data = _padded_buffer([y, u, v], strides, offsets)

    converted = _raw_video_to_bgr_strided(data, 32, 34, "I420", strides, offsets)

    assert converted.shape == original.shape
    assert np.abs(converted.astype(np.int16) - original.astype(np.int16)).mean() < 3


def test_nv12_live_converter_handles_padded_planes():
    original = _solid_bgr((20, 70, 210))
    packed = _nv12_from_bgr(original)
    y = packed[:32, :]
    uv = packed[32:, :]
    strides = [40, 40]
    offsets = [0, 40 * 32 + 9]
    data = _padded_buffer([y, uv], strides, offsets)

    converted = _raw_video_to_bgr_strided(data, 32, 32, "NV12", strides, offsets)

    assert converted.shape == original.shape
    assert np.abs(converted.astype(np.int16) - original.astype(np.int16)).mean() < 3


def test_i420_rejects_missing_plane_metadata():
    data = np.zeros(32 * 32 * 3 // 2, dtype=np.uint8)
    with pytest.raises(FrameConversionError, match="needs 3 planes"):
        _raw_video_to_bgr_strided(data, 32, 32, "I420", [32], [0])


def test_gstreamer_pipeline_uses_selected_transport_and_format():
    tcp = _gstreamer_pipeline("rtsp://example/stream", "h265", "NV12", "tcp")
    udp = _gstreamer_pipeline("rtsp://example/stream", "h264", "BGR", "udp")

    assert "protocols=tcp" in tcp
    assert "mppvideodec format=NV12" in tcp
    assert "protocols=udp" in udp
    assert "mppvideodec format=BGR" in udp


def test_direct_gstreamer_videotestsrc_exercises_live_i420_path():
    pipe = (
        "videotestsrc num-buffers=3 pattern=smpte ! "
        "video/x-raw,format=I420,width=32,height=34,framerate=5/1 ! "
        "appsink name=sink drop=true max-buffers=1 sync=false"
    )
    cap = _GstAppSinkCapture(pipe, "test-i420")
    if not cap.isOpened():
        pytest.skip(cap.last_error or "Python GStreamer unavailable")
    try:
        ok, frame = cap.read()
    finally:
        cap.release()

    assert ok is True
    assert frame is not None
    assert frame.shape == (34, 32, 3)
