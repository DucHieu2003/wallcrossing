from __future__ import annotations

from typing import Literal

from wallcrossing.core.models import Detection

RoiAxis = Literal["auto", "x", "y"]
RoiRect = tuple[int, int, int, int]


def _clamped_span(center: float, extent: float, limit: int) -> tuple[int, int]:
    extent = min(max(1.0, extent), float(limit))
    start = center - extent / 2.0
    end = center + extent / 2.0
    if start < 0:
        end -= start
        start = 0.0
    if end > limit:
        start -= end - limit
        end = float(limit)
    start = max(0.0, start)
    end = min(float(limit), end)
    x1 = int(round(start))
    x2 = int(round(end))
    if x2 <= x1:
        x2 = min(limit, x1 + 1)
        x1 = max(0, x2 - 1)
    return x1, x2


def resolve_roi_axis(
    wall_polygon: list[list[float]],
    axis: RoiAxis,
    frame_shape: tuple[int, int] | None = None,
) -> Literal["x", "y"]:
    if axis != "auto":
        return axis
    xs = [p[0] for p in wall_polygon]
    ys = [p[1] for p in wall_polygon]
    width = max(xs) - min(xs)
    height = max(ys) - min(ys)
    if frame_shape is None:
        return "x" if height >= width else "y"
    frame_h, frame_w = frame_shape
    return "x" if width / frame_w <= height / frame_h else "y"


def roi_from_polygon(
    wall_polygon: list[list[float]],
    frame_shape: tuple[int, int],
    min_extent_ratio: float,
    side_margin_ratio: float,
    axis: RoiAxis = "auto",
) -> RoiRect:
    frame_h, frame_w = frame_shape
    if frame_h <= 0 or frame_w <= 0:
        raise ValueError("frame_shape must be positive")
    if len(wall_polygon) < 3:
        return (0, 0, frame_w, frame_h)

    xs = [p[0] for p in wall_polygon]
    ys = [p[1] for p in wall_polygon]
    poly_x1, poly_x2 = max(0.0, min(xs)), min(float(frame_w), max(xs))
    poly_y1, poly_y2 = max(0.0, min(ys)), min(float(frame_h), max(ys))
    resolved_axis = resolve_roi_axis(wall_polygon, axis, frame_shape)

    if resolved_axis == "x":
        min_extent = frame_w * min_extent_ratio
        polygon_extent = poly_x2 - poly_x1
        extent = polygon_extent
        if polygon_extent < min_extent:
            extent = max(
                min_extent,
                polygon_extent + 2.0 * frame_w * side_margin_ratio,
            )
        center = (poly_x1 + poly_x2) / 2.0
        x1, x2 = _clamped_span(center, extent, frame_w)
        return (x1, 0, x2, frame_h)

    min_extent = frame_h * min_extent_ratio
    polygon_extent = poly_y2 - poly_y1
    extent = polygon_extent
    if polygon_extent < min_extent:
        extent = max(
            min_extent,
            polygon_extent + 2.0 * frame_h * side_margin_ratio,
        )
    center = (poly_y1 + poly_y2) / 2.0
    y1, y2 = _clamped_span(center, extent, frame_h)
    return (0, y1, frame_w, y2)


def translate_detection(det: Detection, dx: int, dy: int) -> Detection:
    x1, y1, x2, y2 = det.bbox_xyxy
    return Detection(
        bbox_xyxy=(x1 + dx, y1 + dy, x2 + dx, y2 + dy),
        confidence=det.confidence,
        class_id=det.class_id,
    )
