from __future__ import annotations

import json
import logging
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from wallcrossing.core.models import Detection

logger = logging.getLogger("wallcrossing.dataset_capture")

_METADATA_MAX_BYTES = 50 * 1024 * 1024


class DatasetCapture:
    """Rate-limited YOLO images/labels with rolling disk usage."""

    def __init__(
        self,
        root_dir: str | Path,
        detection_interval_seconds: float,
        background_interval_seconds: float,
        jpeg_quality: int,
        max_disk_gb: float = 0.0,
        hard_negative_interval_seconds: float = 3600.0,
    ):
        self.root_dir = Path(root_dir)
        self.detection_interval_seconds = detection_interval_seconds
        self.background_interval_seconds = background_interval_seconds
        self.hard_negative_interval_seconds = hard_negative_interval_seconds
        self.jpeg_quality = jpeg_quality
        self.max_disk_bytes = int(max_disk_gb * 1024**3) if max_disk_gb > 0 else 0
        self.metadata_path = self.root_dir / "metadata.jsonl"
        self._last_capture_mono: dict[tuple[str, str], float] = {}
        self._tracked: deque[tuple[Path, Path | None, int]] = deque()
        self._tracked_bytes = 0
        self._scan_existing()
        if self.max_disk_bytes:
            self._enforce_disk_cap()

    def _label_path_for_image(self, image_path: Path) -> Path | None:
        try:
            relative = image_path.relative_to(self.root_dir)
        except ValueError:
            return None
        if not relative.parts or relative.parts[0] != "images":
            return None
        return self.root_dir / "labels" / Path(*relative.parts[1:]).with_suffix(".txt")

    def _scan_existing(self) -> None:
        files: list[tuple[float, Path, Path | None, int]] = []
        newest_by_key: dict[tuple[str, str], float] = {}
        images_root = self.root_dir / "images"
        for image_path in self.root_dir.rglob("*.jpg"):
            try:
                image_stat = image_path.stat()
            except OSError:
                continue
            try:
                relative = image_path.relative_to(images_root)
            except ValueError:
                relative = None
            label_path = self._label_path_for_image(image_path)
            size = image_stat.st_size
            if label_path is not None and label_path.exists():
                try:
                    size += label_path.stat().st_size
                except OSError:
                    pass
            files.append((image_stat.st_mtime, image_path, label_path, size))
            if relative is not None and len(relative.parts) >= 3:
                key = (relative.parts[2], relative.parts[0])
                newest_by_key[key] = max(newest_by_key.get(key, 0.0), image_stat.st_mtime)
        files.sort(key=lambda item: item[0])
        for _, image_path, label_path, size in files:
            self._tracked.append((image_path, label_path, size))
            self._tracked_bytes += size

        now_wall = time.time()
        now_mono = time.monotonic()
        intervals = {
            "detections": self.detection_interval_seconds,
            "background": self.background_interval_seconds,
            "hard_negatives": self.hard_negative_interval_seconds,
        }
        for key, modified_wall in newest_by_key.items():
            remaining = intervals[key[1]] - max(0.0, now_wall - modified_wall)
            if remaining > 0:
                self._last_capture_mono[key] = now_mono - (intervals[key[1]] - remaining)

    def _enforce_disk_cap(self) -> None:
        if not self.max_disk_bytes:
            return
        removed = 0
        while self._tracked_bytes > self.max_disk_bytes and self._tracked:
            image_path, label_path, size = self._tracked.popleft()
            self._tracked_bytes -= size
            for path in (image_path, label_path):
                if path is None:
                    continue
                try:
                    path.unlink()
                except FileNotFoundError:
                    pass
                except OSError:
                    logger.warning("failed to remove old dataset file: %s", path)
            removed += 1
        if removed:
            logger.info(
                "dataset disk cap removed %d oldest image/label pair(s); usage=%.2fGB",
                removed,
                self._tracked_bytes / 1024**3,
            )

    def _rotate_metadata_if_needed(self) -> None:
        try:
            if self.metadata_path.stat().st_size < _METADATA_MAX_BYTES:
                return
        except FileNotFoundError:
            return
        rotated = self.metadata_path.with_suffix(".jsonl.1")
        rotated.unlink(missing_ok=True)
        self.metadata_path.replace(rotated)

    @staticmethod
    def _yolo_lines(
        detections: list[Detection], frame_width: int, frame_height: int
    ) -> list[str]:
        lines: list[str] = []
        for detection in detections:
            x1, y1, x2, y2 = detection.bbox_xyxy
            x1 = max(0.0, min(float(frame_width), x1))
            y1 = max(0.0, min(float(frame_height), y1))
            x2 = max(0.0, min(float(frame_width), x2))
            y2 = max(0.0, min(float(frame_height), y2))
            if x2 <= x1 or y2 <= y1:
                continue
            center_x = ((x1 + x2) / 2.0) / frame_width
            center_y = ((y1 + y2) / 2.0) / frame_height
            width = (x2 - x1) / frame_width
            height = (y2 - y1) / frame_height
            lines.append(
                f"{detection.class_id} {center_x:.6f} {center_y:.6f} "
                f"{width:.6f} {height:.6f}"
            )
        return lines

    def capture(
        self,
        camera_id: str,
        timestamp: str,
        now_mono: float,
        frame: np.ndarray,
        detections: list[Detection],
        overlap_ratios: list[float],
        hard_negative: bool = False,
    ) -> Path | None:
        if detections:
            category = "detections"
            interval = self.detection_interval_seconds
        elif hard_negative:
            category = "hard_negatives"
            interval = self.hard_negative_interval_seconds
        else:
            category = "background"
            interval = self.background_interval_seconds

        rate_key = (camera_id, category)
        last = self._last_capture_mono.get(rate_key)
        if last is not None and now_mono - last < interval:
            return None

        safe_stamp = (
            timestamp.replace(":", "").replace("-", "").replace(".", "").replace("+", "Z")
        )
        stem = f"{safe_stamp}_{camera_id}"
        relative_dir = Path(category) / timestamp[:10] / camera_id
        image_path = self.root_dir / "images" / relative_dir / f"{stem}.jpg"
        label_path = self.root_dir / "labels" / relative_dir / f"{stem}.txt"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.parent.mkdir(parents=True, exist_ok=True)

        written = cv2.imwrite(
            str(image_path),
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality],
        )
        if not written:
            raise OSError(f"failed to write dataset frame: {image_path}")

        frame_height, frame_width = frame.shape[:2]
        label_lines = self._yolo_lines(detections, frame_width, frame_height)
        try:
            label_path.write_text(
                "\n".join(label_lines) + ("\n" if label_lines else ""),
                encoding="utf-8",
            )
        except OSError:
            image_path.unlink(missing_ok=True)
            raise

        if self.max_disk_bytes:
            try:
                pair_size = image_path.stat().st_size + label_path.stat().st_size
            except OSError:
                pair_size = 0
            self._tracked.append((image_path, label_path, pair_size))
            self._tracked_bytes += pair_size
            self._enforce_disk_cap()

        record = {
            "camera_id": camera_id,
            "timestamp": timestamp,
            "category": category,
            "image_path": str(image_path),
            "label_path": str(label_path),
            "frame_width": frame_width,
            "frame_height": frame_height,
            "detections": [
                {
                    "bbox_xyxy": [round(v, 1) for v in detection.bbox_xyxy],
                    "confidence": round(detection.confidence, 4),
                    "class_id": detection.class_id,
                    "wall_overlap_ratio": round(overlap, 4),
                }
                for detection, overlap in zip(detections, overlap_ratios)
            ],
        }
        self.metadata_path.parent.mkdir(parents=True, exist_ok=True)
        self._rotate_metadata_if_needed()
        with self.metadata_path.open("a", encoding="utf-8") as metadata_file:
            metadata_file.write(json.dumps(record, ensure_ascii=False) + "\n")

        self._last_capture_mono[rate_key] = now_mono
        logger.info(
            "dataset %s cam=%s labels=%d -> %s",
            category,
            camera_id,
            len(label_lines),
            image_path,
        )
        return image_path
