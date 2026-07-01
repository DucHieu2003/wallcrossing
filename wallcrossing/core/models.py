from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class Detection:
    bbox_xyxy: tuple[float, float, float, float]
    confidence: float
    class_id: int


@dataclass
class AlertEvent:
    alert_id: str
    camera_id: str
    camera_name: str
    timestamp: str
    bbox_xyxy: tuple[float, float, float, float]
    confidence: float
    wall_polygon: list[list[float]]
    overlap_ratio: float
    evidence_path: Optional[str] = None

    def to_log_dict(self) -> dict:
        return {
            "alert_id": self.alert_id,
            "camera_id": self.camera_id,
            "camera_name": self.camera_name,
            "timestamp": self.timestamp,
            "bbox_xyxy": [round(v, 1) for v in self.bbox_xyxy],
            "confidence": round(self.confidence, 4),
            "wall_polygon": self.wall_polygon,
            "overlap_ratio": round(self.overlap_ratio, 4),
            "evidence_path": self.evidence_path,
        }
