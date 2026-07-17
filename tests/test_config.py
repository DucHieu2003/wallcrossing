import pytest

from wallcrossing.core.config_loader import AppConfig, load_runtime_config


def _valid_raw():
    return {
        "cameras": [
            {
                "id": "cam_001",
                "name": "Gate 1",
                "rtsp_url": "rtsp://x/stream",
                "wall_polygon": [[0, 0], [10, 0], [10, 10], [0, 10]],
            }
        ]
    }


def test_valid_config_loads():
    cfg = AppConfig.model_validate(_valid_raw())
    assert len(cfg.cameras) == 1
    assert cfg.model.backend == "rknn"
    assert cfg.rules.contact_mode == "bottom_band"
    assert cfg.pipeline.detect_roi_enabled is False
    assert cfg.pipeline.rtsp_transport == "tcp"
    assert cfg.pipeline.shutdown_timeout_seconds == 15
    assert cfg.pipeline.health_log_interval_seconds == 60
    assert cfg.pipeline.stale_reader_warn_seconds == 60
    assert cfg.pipeline.max_consecutive_process_errors == 30
    assert cfg.pipeline.dataset_capture_enabled is False
    assert cfg.pipeline.dataset_detection_interval_seconds == 5
    assert cfg.pipeline.dataset_background_interval_seconds == 3600
    assert cfg.pipeline.dataset_jpeg_quality == 90


def test_duplicate_camera_ids_rejected():
    raw = _valid_raw()
    raw["cameras"].append(dict(raw["cameras"][0]))
    with pytest.raises(ValueError, match="duplicate camera ids"):
        AppConfig.model_validate(raw)


def test_polygon_too_few_points_rejected():
    raw = _valid_raw()
    raw["cameras"][0]["wall_polygon"] = [[0, 0], [1, 1]]
    with pytest.raises(ValueError, match="at least 3 points"):
        AppConfig.model_validate(raw)


def test_bad_confidence_rejected():
    raw = _valid_raw()
    raw["model"] = {"confidence": 1.5}
    with pytest.raises(ValueError):
        AppConfig.model_validate(raw)


def test_bad_imgsz_rejected():
    raw = _valid_raw()
    raw["model"] = {"imgsz": 641}
    with pytest.raises(ValueError, match="multiple of 32"):
        AppConfig.model_validate(raw)


def test_detect_fps_fallback_to_default():
    raw = _valid_raw()
    raw["pipeline"] = {"default_detect_fps": 2.0}
    cfg = AppConfig.model_validate(raw)
    assert cfg.detect_fps_for(cfg.cameras[0]) == 2.0


def test_roi_config_loads_and_camera_override_wins():
    raw = _valid_raw()
    raw["pipeline"] = {
        "detect_roi_enabled": True,
        "detect_roi_axis": "auto",
        "detect_roi_min_extent_ratio": 0.5,
        "detect_roi_side_margin_ratio": 0.05,
    }
    raw["cameras"][0]["detect_roi_enabled"] = False
    raw["cameras"][0]["detect_roi_axis"] = "x"
    raw["cameras"][0]["detect_roi_min_extent_ratio"] = 0.6
    raw["cameras"][0]["detect_roi_side_margin_ratio"] = 0.08

    cfg = AppConfig.model_validate(raw)
    cam = cfg.cameras[0]

    assert cfg.detect_roi_enabled_for(cam) is False
    assert cfg.detect_roi_axis_for(cam) == "x"
    assert cfg.detect_roi_min_extent_ratio_for(cam) == 0.6
    assert cfg.detect_roi_side_margin_ratio_for(cam) == 0.08


def test_bad_runtime_controls_rejected():
    raw = _valid_raw()
    raw["pipeline"] = {"shutdown_timeout_seconds": 0}
    with pytest.raises(ValueError, match="runtime timeouts"):
        AppConfig.model_validate(raw)

    raw = _valid_raw()
    raw["pipeline"] = {"max_consecutive_process_errors": 0}
    with pytest.raises(ValueError, match="max_consecutive_process_errors"):
        AppConfig.model_validate(raw)


def test_bad_roi_ratio_rejected():
    raw = _valid_raw()
    raw["pipeline"] = {"detect_roi_min_extent_ratio": 0.0}
    with pytest.raises(ValueError, match="detect_roi_min_extent_ratio"):
        AppConfig.model_validate(raw)

    raw = _valid_raw()
    raw["cameras"][0]["detect_roi_side_margin_ratio"] = -0.1
    with pytest.raises(ValueError, match="detect_roi_side_margin_ratio"):
        AppConfig.model_validate(raw)


def test_runtime_config_loads_from_root_config():
    cfg = load_runtime_config()
    assert len(cfg.cameras) > 0
    assert cfg.pipeline.default_detect_fps == 5
    assert cfg.pipeline.rtsp_transport == "tcp"
    assert cfg.pipeline.detect_roi_enabled is False
    assert cfg.pipeline.shutdown_timeout_seconds == 15
    assert cfg.pipeline.health_log_interval_seconds == 60
    assert cfg.pipeline.stale_reader_warn_seconds == 60
    assert cfg.pipeline.max_consecutive_process_errors == 30
    assert cfg.pipeline.dataset_capture_enabled is True
    assert cfg.pipeline.dataset_capture_dir.endswith("outputs/dataset_capture")
    assert cfg.pipeline.dataset_detection_interval_seconds == 5
    assert cfg.pipeline.dataset_background_interval_seconds == 28800
    assert cfg.pipeline.dataset_hard_negative_interval_seconds == 21600
    assert cfg.pipeline.dataset_jpeg_quality == 90
    assert cfg.pipeline.dataset_max_disk_gb == 20.0
    assert cfg.pipeline.motion_filter_enabled is True
    assert cfg.pipeline.motion_min_ratio == 0.05
    assert cfg.pipeline.alerts_enabled is True
    assert cfg.pipeline.evidence_max_disk_gb == 2.0
    assert cfg.pipeline.debug_preview_max_disk_gb == 1.0
    assert cfg.pipeline.alert_log_max_mb == 10
    assert cfg.pipeline.alert_log_backup_count == 2
    assert cfg.pipeline.rss_graceful_restart_mb == 4800
    assert cfg.pipeline.rss_check_interval_seconds == 5.0
