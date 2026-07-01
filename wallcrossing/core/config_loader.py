from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class ModelConfig(BaseModel):
    rknn_path: str = "weights/yolo26s_rk3588_fp16.rknn"
    pt_path: str = "weights/yolo26s.pt"
    backend: Literal["rknn", "ultralytics", "mock"] = "rknn"
    npu_cores: list[int] = Field(default_factory=lambda: [0, 1, 2])
    imgsz: int = 640
    confidence: float = 0.45
    person_class_id: int = 0

    @field_validator("confidence")
    @classmethod
    def _conf_range(cls, v: float) -> float:
        if not 0.0 < v < 1.0:
            raise ValueError("confidence must be in (0, 1)")
        return v

    @field_validator("imgsz")
    @classmethod
    def _imgsz_positive(cls, v: int) -> int:
        if v <= 0 or v % 32 != 0:
            raise ValueError("imgsz must be a positive multiple of 32")
        return v


class PipelineConfig(BaseModel):
    default_detect_fps: float = 0.0
    decode_backend: Literal["gstreamer", "opencv"] = "gstreamer"
    evidence_dir: str = "outputs/evidence"
    alert_log_path: str = "logs/alerts.jsonl"

    @field_validator("default_detect_fps")
    @classmethod
    def _fps_nonneg(cls, v: float) -> float:
        if v < 0:
            raise ValueError("detect_fps must be >= 0 (0 = max speed)")
        return v


class RulesConfig(BaseModel):
    min_overlap_ratio: float = 0.02
    consecutive_hits: int = 2
    cooldown_seconds: float = 30.0
    contact_mode: Literal["full_bbox", "bottom_band"] = "bottom_band"
    bottom_band_ratio: float = 0.25

    @field_validator("min_overlap_ratio", "bottom_band_ratio")
    @classmethod
    def _ratio_range(cls, v: float) -> float:
        if not 0.0 < v <= 1.0:
            raise ValueError("ratio must be in (0, 1]")
        return v

    @field_validator("consecutive_hits")
    @classmethod
    def _hits_positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError("consecutive_hits must be >= 1")
        return v


class CameraConfig(BaseModel):
    id: str
    name: str = ""
    rtsp_url: str
    enabled: bool = True
    detect_fps: float | None = None
    wall_polygon: list[list[float]]

    @field_validator("wall_polygon")
    @classmethod
    def _polygon_valid(cls, v: list[list[float]]) -> list[list[float]]:
        if len(v) < 3:
            raise ValueError("wall_polygon needs at least 3 points")
        for pt in v:
            if len(pt) != 2:
                raise ValueError(f"polygon point must be [x, y], got {pt}")
        return v


class AppConfig(BaseModel):
    model: ModelConfig = Field(default_factory=ModelConfig)
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)
    rules: RulesConfig = Field(default_factory=RulesConfig)
    cameras: list[CameraConfig]

    @model_validator(mode="after")
    def _unique_camera_ids(self) -> "AppConfig":
        ids = [c.id for c in self.cameras]
        dupes = {i for i in ids if ids.count(i) > 1}
        if dupes:
            raise ValueError(f"duplicate camera ids: {sorted(dupes)}")
        return self

    def enabled_cameras(self) -> list[CameraConfig]:
        return [c for c in self.cameras if c.enabled]

    def detect_fps_for(self, cam: CameraConfig) -> float:
        return cam.detect_fps if cam.detect_fps is not None else self.pipeline.default_detect_fps


def load_runtime_config() -> AppConfig:
    """Load config from root config.py, matching the reference service style."""
    import config as runtime_config

    raw = {
        "model": {
            "rknn_path": runtime_config.YOLO26_RKNN_PATH,
            "pt_path": runtime_config.YOLO26_PT_PATH,
            "backend": runtime_config.MODEL_BACKEND,
            "npu_cores": runtime_config.NPU_CORES,
            "imgsz": runtime_config.IMG_SIZE,
            "confidence": runtime_config.DET_CONF_THRES,
            "person_class_id": runtime_config.PERSON_CLASS_ID,
        },
        "pipeline": {
            "default_detect_fps": runtime_config.DEFAULT_DETECT_FPS,
            "decode_backend": runtime_config.DECODE_BACKEND,
            "evidence_dir": runtime_config.EVIDENCE_DIR,
            "alert_log_path": runtime_config.ALERT_LOG_PATH,
        },
        "rules": {
            "min_overlap_ratio": runtime_config.MIN_OVERLAP_RATIO,
            "consecutive_hits": runtime_config.CONSECUTIVE_HITS,
            "cooldown_seconds": runtime_config.COOLDOWN_SECONDS,
            "contact_mode": runtime_config.CONTACT_MODE,
            "bottom_band_ratio": runtime_config.BOTTOM_BAND_RATIO,
        },
        "cameras": runtime_config.CAMERA_CONFIGS,
    }
    return AppConfig.model_validate(raw)
