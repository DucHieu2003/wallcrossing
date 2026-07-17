from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ModelConfig(BaseModel):
    rknn_path: str = "weights/yolo26s_rk3588_fp16.rknn"
    backend: Literal["rknn", "mock"] = "rknn"
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
    codec: Literal["h264", "h265"] = "h264"
    rtsp_transport: Literal["tcp", "udp"] = "tcp"
    ffmpeg_video_codec: str = ""
    evidence_dir: str = "outputs/evidence"
    evidence_max_disk_gb: float = 0.0
    alert_log_path: str = "logs/alerts.jsonl"
    alert_log_max_mb: int = 10
    alert_log_backup_count: int = 2
    alerts_enabled: bool = True
    debug_preview_enabled: bool = False
    debug_preview_dir: str = "outputs/debug_preview"
    debug_preview_max_disk_gb: float = 0.0
    debug_preview_interval_seconds: float = 2.0
    dataset_capture_enabled: bool = False
    dataset_capture_dir: str = "outputs/dataset_capture"
    dataset_detection_interval_seconds: float = 5.0
    dataset_background_interval_seconds: float = 3600.0
    dataset_hard_negative_interval_seconds: float = 3600.0
    dataset_jpeg_quality: int = 90
    dataset_max_disk_gb: float = 0.0
    motion_filter_enabled: bool = False
    motion_min_ratio: float = 0.05
    detect_roi_enabled: bool = False
    detect_roi_axis: Literal["auto", "x", "y"] = "auto"
    detect_roi_min_extent_ratio: float = 0.5
    detect_roi_side_margin_ratio: float = 0.05
    shutdown_timeout_seconds: float = 15.0
    health_log_interval_seconds: float = 60.0
    stale_reader_warn_seconds: float = 60.0
    max_consecutive_process_errors: int = 30
    rss_graceful_restart_mb: float = 0.0
    rss_check_interval_seconds: float = 5.0

    @field_validator("default_detect_fps")
    @classmethod
    def _fps_nonneg(cls, v: float) -> float:
        if v < 0:
            raise ValueError("detect_fps must be >= 0 (0 = max speed)")
        return v

    @field_validator(
        "shutdown_timeout_seconds",
        "health_log_interval_seconds",
        "stale_reader_warn_seconds",
        "dataset_detection_interval_seconds",
        "dataset_background_interval_seconds",
        "dataset_hard_negative_interval_seconds",
        "rss_check_interval_seconds",
    )
    @classmethod
    def _timeout_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("runtime timeouts must be > 0")
        return v

    @field_validator("dataset_jpeg_quality")
    @classmethod
    def _jpeg_quality_range(cls, v: int) -> int:
        if not 1 <= v <= 100:
            raise ValueError("dataset_jpeg_quality must be in [1, 100]")
        return v

    @field_validator(
        "dataset_max_disk_gb",
        "evidence_max_disk_gb",
        "debug_preview_max_disk_gb",
        "rss_graceful_restart_mb",
    )
    @classmethod
    def _storage_and_memory_nonnegative(cls, v: float) -> float:
        if v < 0:
            raise ValueError("storage quotas and RSS threshold must be >= 0")
        return v

    @field_validator("alert_log_max_mb", "alert_log_backup_count")
    @classmethod
    def _alert_rotation_nonnegative(cls, v: int) -> int:
        if v < 0:
            raise ValueError("alert log rotation values must be >= 0")
        return v

    @field_validator("motion_min_ratio")
    @classmethod
    def _motion_ratio_range(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("motion_min_ratio must be in [0, 1]")
        return v

    @field_validator("max_consecutive_process_errors")
    @classmethod
    def _max_process_errors_positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError("max_consecutive_process_errors must be >= 1")
        return v

    @field_validator("detect_roi_min_extent_ratio")
    @classmethod
    def _roi_min_extent_range(cls, v: float) -> float:
        if not 0.0 < v <= 1.0:
            raise ValueError("detect_roi_min_extent_ratio must be in (0, 1]")
        return v

    @field_validator("detect_roi_side_margin_ratio")
    @classmethod
    def _roi_side_margin_range(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("detect_roi_side_margin_ratio must be in [0, 1]")
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
    # (W, H) of the image the polygon was drawn on; None = already in frame coords
    polygon_ref_size: list[float] | None = None
    detect_roi_enabled: bool | None = None
    detect_roi_axis: Literal["auto", "x", "y"] | None = None
    detect_roi_min_extent_ratio: float | None = None
    detect_roi_side_margin_ratio: float | None = None

    @field_validator("polygon_ref_size")
    @classmethod
    def _ref_size_valid(cls, v: list[float] | None) -> list[float] | None:
        if v is not None and (len(v) != 2 or v[0] <= 0 or v[1] <= 0):
            raise ValueError("polygon_ref_size must be [width, height] > 0")
        return v

    @field_validator("wall_polygon")
    @classmethod
    def _polygon_valid(cls, v: list[list[float]]) -> list[list[float]]:
        if len(v) < 3:
            raise ValueError("wall_polygon needs at least 3 points")
        for pt in v:
            if len(pt) != 2:
                raise ValueError(f"polygon point must be [x, y], got {pt}")
        return v

    @field_validator("detect_roi_min_extent_ratio")
    @classmethod
    def _roi_min_extent_range(cls, v: float | None) -> float | None:
        if v is not None and not 0.0 < v <= 1.0:
            raise ValueError("detect_roi_min_extent_ratio must be in (0, 1]")
        return v

    @field_validator("detect_roi_side_margin_ratio")
    @classmethod
    def _roi_side_margin_range(cls, v: float | None) -> float | None:
        if v is not None and not 0.0 <= v <= 1.0:
            raise ValueError("detect_roi_side_margin_ratio must be in [0, 1]")
        return v


class AppConfig(BaseModel):
    # Tat canh bao namespace bao ve "model_" cua Pydantic v2 vi co field ten "model"
    model_config = ConfigDict(protected_namespaces=())

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

    def detect_roi_enabled_for(self, cam: CameraConfig) -> bool:
        return cam.detect_roi_enabled if cam.detect_roi_enabled is not None else self.pipeline.detect_roi_enabled

    def detect_roi_axis_for(self, cam: CameraConfig) -> Literal["auto", "x", "y"]:
        return cam.detect_roi_axis or self.pipeline.detect_roi_axis

    def detect_roi_min_extent_ratio_for(self, cam: CameraConfig) -> float:
        return (
            cam.detect_roi_min_extent_ratio
            if cam.detect_roi_min_extent_ratio is not None
            else self.pipeline.detect_roi_min_extent_ratio
        )

    def detect_roi_side_margin_ratio_for(self, cam: CameraConfig) -> float:
        return (
            cam.detect_roi_side_margin_ratio
            if cam.detect_roi_side_margin_ratio is not None
            else self.pipeline.detect_roi_side_margin_ratio
        )


def load_runtime_config() -> AppConfig:
    """Load config from root config.py, matching the reference service style."""
    import config as runtime_config

    raw = {
        "model": {
            "rknn_path": runtime_config.YOLO26_RKNN_PATH,
            "backend": runtime_config.MODEL_BACKEND,
            "npu_cores": runtime_config.NPU_CORES,
            "imgsz": runtime_config.IMG_SIZE,
            "confidence": runtime_config.DET_CONF_THRES,
            "person_class_id": runtime_config.PERSON_CLASS_ID,
        },
        "pipeline": {
            "default_detect_fps": runtime_config.DEFAULT_DETECT_FPS,
            "decode_backend": runtime_config.DECODE_BACKEND,
            "codec": getattr(runtime_config, "RTSP_CODEC", "h264"),
            "rtsp_transport": getattr(runtime_config, "RTSP_TRANSPORT", "tcp"),
            "ffmpeg_video_codec": getattr(runtime_config, "FFMPEG_VIDEO_CODEC", ""),
            "evidence_dir": runtime_config.EVIDENCE_DIR,
            "evidence_max_disk_gb": getattr(
                runtime_config, "EVIDENCE_MAX_DISK_GB", 0.0
            ),
            "alert_log_path": runtime_config.ALERT_LOG_PATH,
            "alert_log_max_mb": getattr(runtime_config, "ALERT_LOG_MAX_MB", 10),
            "alert_log_backup_count": getattr(
                runtime_config, "ALERT_LOG_BACKUP_COUNT", 2
            ),
            "alerts_enabled": getattr(runtime_config, "ALERTS_ENABLED", True),
            "debug_preview_enabled": getattr(
                runtime_config, "DEBUG_PREVIEW_ENABLED", False
            ),
            "debug_preview_dir": getattr(
                runtime_config, "DEBUG_PREVIEW_DIR", "outputs/debug_preview"
            ),
            "debug_preview_max_disk_gb": getattr(
                runtime_config, "DEBUG_PREVIEW_MAX_DISK_GB", 0.0
            ),
            "debug_preview_interval_seconds": getattr(
                runtime_config, "DEBUG_PREVIEW_INTERVAL_SECONDS", 2.0
            ),
            "dataset_capture_enabled": getattr(
                runtime_config, "DATASET_CAPTURE_ENABLED", False
            ),
            "dataset_capture_dir": getattr(
                runtime_config, "DATASET_CAPTURE_DIR", "outputs/dataset_capture"
            ),
            "dataset_detection_interval_seconds": getattr(
                runtime_config, "DATASET_DETECTION_INTERVAL_SECONDS", 5.0
            ),
            "dataset_background_interval_seconds": getattr(
                runtime_config, "DATASET_BACKGROUND_INTERVAL_SECONDS", 3600.0
            ),
            "dataset_hard_negative_interval_seconds": getattr(
                runtime_config, "DATASET_HARD_NEGATIVE_INTERVAL_SECONDS", 3600.0
            ),
            "dataset_jpeg_quality": getattr(
                runtime_config, "DATASET_JPEG_QUALITY", 90
            ),
            "dataset_max_disk_gb": getattr(
                runtime_config, "DATASET_MAX_DISK_GB", 0.0
            ),
            "motion_filter_enabled": getattr(
                runtime_config, "MOTION_FILTER_ENABLED", False
            ),
            "motion_min_ratio": getattr(runtime_config, "MOTION_MIN_RATIO", 0.05),
            "detect_roi_enabled": getattr(runtime_config, "DETECT_ROI_ENABLED", False),
            "detect_roi_axis": getattr(runtime_config, "DETECT_ROI_AXIS", "auto"),
            "detect_roi_min_extent_ratio": getattr(
                runtime_config, "DETECT_ROI_MIN_EXTENT_RATIO", 0.5
            ),
            "detect_roi_side_margin_ratio": getattr(
                runtime_config, "DETECT_ROI_SIDE_MARGIN_RATIO", 0.05
            ),
            "shutdown_timeout_seconds": getattr(
                runtime_config, "SHUTDOWN_TIMEOUT_SECONDS", 15
            ),
            "health_log_interval_seconds": getattr(
                runtime_config, "HEALTH_LOG_INTERVAL_SECONDS", 60
            ),
            "stale_reader_warn_seconds": getattr(
                runtime_config, "STALE_READER_WARN_SECONDS", 60
            ),
            "max_consecutive_process_errors": getattr(
                runtime_config, "MAX_CONSECUTIVE_PROCESS_ERRORS", 30
            ),
            "rss_graceful_restart_mb": getattr(
                runtime_config, "RSS_GRACEFUL_RESTART_MB", 0.0
            ),
            "rss_check_interval_seconds": getattr(
                runtime_config, "RSS_CHECK_INTERVAL_SECONDS", 5.0
            ),
        },
        "rules": {
            "min_overlap_ratio": runtime_config.MIN_OVERLAP_RATIO,
            "consecutive_hits": runtime_config.CONSECUTIVE_HITS,
            "cooldown_seconds": runtime_config.COOLDOWN_SECONDS,
            "contact_mode": runtime_config.CONTACT_MODE,
            "bottom_band_ratio": runtime_config.BOTTOM_BAND_RATIO,
        },
        "cameras": [
            {"polygon_ref_size": getattr(runtime_config, "POLYGON_REF_SIZE", None), **cam}
            for cam in runtime_config.CAMERA_CONFIGS
        ],
    }
    return AppConfig.model_validate(raw)
