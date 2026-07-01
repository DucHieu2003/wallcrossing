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


def test_runtime_config_loads_from_root_config():
    cfg = load_runtime_config()
    assert len(cfg.cameras) > 0
    assert cfg.pipeline.default_detect_fps == 5
