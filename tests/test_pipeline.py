import numpy as np

from wallcrossing.core.config_loader import AppConfig
from wallcrossing.core.models import Detection
from wallcrossing.runtime import pipeline as pipeline_module
from wallcrossing.runtime.pipeline import Pipeline


class FakeReader:
    def __init__(self, frame):
        self.frame = frame
        self.index = 1

    def read_latest(self):
        return self.frame, self.index

    def request_stop(self):
        pass

    def join(self, timeout=None):
        del timeout
        return True


class RecordingDetector:
    def __init__(self):
        self.input_shape = None

    def detect(self, image):
        self.input_shape = image.shape
        return [Detection(bbox_xyxy=(200, 100, 300, 400), confidence=0.9, class_id=0)]

    def close(self):
        pass


def _cfg(tmp_path):
    return AppConfig.model_validate(
        {
            "model": {"backend": "mock"},
            "pipeline": {
                "detect_roi_enabled": True,
                "detect_roi_min_extent_ratio": 0.5,
                "detect_roi_side_margin_ratio": 0.05,
                "evidence_dir": str(tmp_path / "evidence"),
                "alert_log_path": str(tmp_path / "alerts.jsonl"),
            },
            "rules": {"consecutive_hits": 1, "cooldown_seconds": 0.0},
            "cameras": [
                {
                    "id": "cam_001",
                    "rtsp_url": "rtsp://x/stream",
                    "wall_polygon": [[450, 0], [550, 0], [550, 500], [450, 500]],
                }
            ],
        }
    )


def test_pipeline_detects_on_roi_and_offsets_bbox_before_evidence(tmp_path, monkeypatch):
    frame = np.zeros((500, 1000, 3), dtype=np.uint8)
    cfg = _cfg(tmp_path)
    pipeline = Pipeline(cfg)
    detector = RecordingDetector()
    captured = {}

    def fake_draw_and_save(image, wall_polygon, detection, label, out_path):
        captured["image_shape"] = image.shape
        captured["bbox"] = detection.bbox_xyxy

    pipeline.detector = detector
    pipeline.readers = {"cam_001": FakeReader(frame)}
    monkeypatch.setattr(pipeline_module, "draw_and_save", fake_draw_and_save)

    pipeline._process_camera("cam_001", now_mono=1.0)

    assert detector.input_shape == (500, 500, 3)
    assert captured["image_shape"] == frame.shape
    assert captured["bbox"] == (450, 100, 550, 400)


def test_pipeline_passes_raw_frame_and_all_detections_to_dataset_capture(tmp_path):
    frame = np.zeros((500, 1000, 3), dtype=np.uint8)
    cfg = _cfg(tmp_path)
    pipeline = Pipeline(cfg)
    detector = RecordingDetector()
    captured = {}

    class RecordingCapture:
        def capture(self, **kwargs):
            captured.update(kwargs)

    pipeline.detector = detector
    pipeline.readers = {"cam_001": FakeReader(frame)}
    pipeline.dataset_capture = RecordingCapture()

    pipeline._process_camera("cam_001", now_mono=7.0)

    assert captured["frame"] is frame
    assert captured["now_mono"] == 7.0
    assert captured["detections"][0].bbox_xyxy == (450, 100, 550, 400)
    assert len(captured["overlap_ratios"]) == 1


def test_pipeline_marks_static_model_detection_as_hard_negative(tmp_path):
    frame = np.zeros((500, 1000, 3), dtype=np.uint8)
    cfg = _cfg(tmp_path)
    pipeline = Pipeline(cfg)
    captured = {}

    class StaticMotionFilter:
        def filter(self, camera_id, image, detections, overlaps):
            del camera_id, image, detections, overlaps
            return [], []

        def is_ready(self, camera_id):
            del camera_id
            return True

    class RecordingCapture:
        def capture(self, **kwargs):
            captured.update(kwargs)

    pipeline.detector = RecordingDetector()
    pipeline.readers = {"cam_001": FakeReader(frame)}
    pipeline.motion_filter = StaticMotionFilter()
    pipeline.dataset_capture = RecordingCapture()

    pipeline._process_camera("cam_001", now_mono=7.0)

    assert captured["detections"] == []
    assert captured["hard_negative"] is True


def test_pipeline_does_not_capture_detections_during_motion_warmup(tmp_path):
    frame = np.zeros((500, 1000, 3), dtype=np.uint8)
    cfg = _cfg(tmp_path)
    pipeline = Pipeline(cfg)

    class WarmingMotionFilter:
        def filter(self, camera_id, image, detections, overlaps):
            del camera_id, image, detections, overlaps
            return [], []

        def is_ready(self, camera_id):
            del camera_id
            return False

    class FailCapture:
        def capture(self, **kwargs):
            raise AssertionError(f"capture should be skipped during warmup: {kwargs}")

    pipeline.detector = RecordingDetector()
    pipeline.readers = {"cam_001": FakeReader(frame)}
    pipeline.motion_filter = WarmingMotionFilter()
    pipeline.dataset_capture = FailCapture()

    pipeline._process_camera("cam_001", now_mono=7.0)


def test_pipeline_requests_restart_when_rss_reaches_threshold(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    cfg.pipeline.rss_graceful_restart_mb = 4800
    cfg.pipeline.rss_check_interval_seconds = 5
    pipeline = Pipeline(cfg)
    pipeline._next_rss_check_mono = 0.0
    monkeypatch.setattr(pipeline_module, "_process_memory_mb", lambda: (4900.0, 5000.0))

    pipeline._check_memory(now_mono=10.0)

    assert pipeline._stop is True


def test_pipeline_below_rss_threshold_keeps_running(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    cfg.pipeline.rss_graceful_restart_mb = 4800
    pipeline = Pipeline(cfg)
    pipeline._next_rss_check_mono = 0.0
    monkeypatch.setattr(pipeline_module, "_process_memory_mb", lambda: (4700.0, 4800.0))

    pipeline._check_memory(now_mono=10.0)

    assert pipeline._stop is False


def test_pipeline_stop_is_idempotent(tmp_path):
    pipeline = Pipeline(_cfg(tmp_path))
    reader = FakeReader(np.zeros((10, 10, 3), dtype=np.uint8))
    pipeline.readers = {"cam_001": reader}
    detector = RecordingDetector()
    detector.close_count = 0

    def close():
        detector.close_count += 1

    detector.close = close
    pipeline.detector = detector

    pipeline.stop()
    pipeline.stop()

    assert detector.close_count == 1


def test_pipeline_sends_systemd_lifecycle_notifications(tmp_path):
    notifications = []
    pipeline = Pipeline(_cfg(tmp_path), notify_systemd=notifications.append)

    pipeline.start()
    pipeline._notify_watchdog()
    pipeline.stop()

    assert notifications == ["READY=1", "WATCHDOG=1", "STOPPING=1"]
