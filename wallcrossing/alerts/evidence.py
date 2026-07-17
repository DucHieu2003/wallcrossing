from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from wallcrossing.core.models import Detection


def _date_dir(timestamp: str) -> str:
    # timestamp is ISO-8601, date part is the first 10 chars (YYYY-MM-DD)
    return timestamp[:10]


def _safe_stamp(timestamp: str) -> str:
    return timestamp.replace(":", "").replace("-", "").replace(".", "").replace("+", "Z")


def evidence_path(
    evidence_dir: str | Path,
    camera_id: str,
    timestamp: str,
    alert_id: str,
) -> Path:
    return (
        Path(evidence_dir)
        / _date_dir(timestamp)
        / camera_id
        / f"{_safe_stamp(timestamp)}_{alert_id}.jpg"
    )


def debug_preview_path(
    preview_dir: str | Path,
    camera_id: str,
    timestamp: str,
) -> Path:
    return (
        Path(preview_dir)
        / _date_dir(timestamp)
        / camera_id
        / f"{_safe_stamp(timestamp)}_{camera_id}.jpg"
    )


def draw_detections_preview(
    image: np.ndarray,
    wall_polygon: list[list[float]],
    detections: list[Detection],
    label: str,
    out_path: Path,
) -> None:
    canvas = image.copy()

    pts = np.array(wall_polygon, dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(canvas, [pts], isClosed=True, color=(0, 200, 0), thickness=2)

    for det in detections:
        x1, y1, x2, y2 = (int(v) for v in det.bbox_xyxy)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)
        conf = f"{det.confidence:.2f}"
        (tw, th), _ = cv2.getTextSize(conf, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        ly = max(0, y1 - th - 4)
        cv2.rectangle(canvas, (x1, ly), (x1 + tw + 4, ly + th + 4), (0, 0, 255), -1)
        cv2.putText(
            canvas,
            conf,
            (x1 + 2, ly + th + 1),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    cv2.putText(
        canvas, label, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), canvas)


def draw_and_save(
    image: np.ndarray,
    wall_polygon: list[list[float]],
    detection: Detection,
    label: str,
    out_path: Path,
) -> None:
    canvas = image.copy()

    pts = np.array(wall_polygon, dtype=np.int32).reshape((-1, 1, 2))
    overlay = canvas.copy()
    cv2.fillPoly(overlay, [pts], color=(0, 200, 0))
    cv2.addWeighted(overlay, 0.25, canvas, 0.75, 0, canvas)
    cv2.polylines(canvas, [pts], isClosed=True, color=(0, 200, 0), thickness=2)

    x1, y1, x2, y2 = (int(v) for v in detection.bbox_xyxy)
    cv2.rectangle(canvas, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)

    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    ly = max(0, y1 - th - 6)
    cv2.rectangle(canvas, (x1, ly), (x1 + tw + 6, ly + th + 6), (0, 0, 255), -1)
    cv2.putText(
        canvas,
        label,
        (x1 + 3, ly + th + 1),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), canvas)
