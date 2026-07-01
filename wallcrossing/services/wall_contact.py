from __future__ import annotations

import numpy as np


def _polygon_area(poly: np.ndarray) -> float:
    x = poly[:, 0]
    y = poly[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def _clip_polygon(subject: np.ndarray, clip: np.ndarray) -> np.ndarray:
    """Sutherland-Hodgman: clip `subject` polygon against convex-ish `clip` polygon.

    Works correctly when `clip` (the bbox rectangle here) is convex.
    Returns the clipped polygon vertices (possibly empty).
    """
    output = subject

    n = len(clip)
    for i in range(n):
        a = clip[i]
        b = clip[(i + 1) % n]
        edge = b - a
        if len(output) == 0:
            break
        input_list = output
        output = []

        def inside(p: np.ndarray) -> float:
            # signed cross product; sign depends on clip winding
            return edge[0] * (p[1] - a[1]) - edge[1] * (p[0] - a[0])

        for j in range(len(input_list)):
            cur = input_list[j]
            prev = input_list[j - 1]
            cur_in = inside(cur)
            prev_in = inside(prev)
            if cur_in >= 0:
                if prev_in < 0:
                    output.append(_intersect(prev, cur, a, b))
                output.append(cur)
            elif prev_in >= 0:
                output.append(_intersect(prev, cur, a, b))

    return np.array(output, dtype=float) if len(output) else np.empty((0, 2))


def _intersect(p1: np.ndarray, p2: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    r = p2 - p1
    s = b - a
    denom = r[0] * s[1] - r[1] * s[0]
    if abs(denom) < 1e-12:
        return p1
    t = ((a[0] - p1[0]) * s[1] - (a[1] - p1[1]) * s[0]) / denom
    return p1 + t * r


def _ensure_ccw(poly: np.ndarray) -> np.ndarray:
    x = poly[:, 0]
    y = poly[:, 1]
    signed = np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))
    # signed > 0 => clockwise in image coords (y-down); flip to get consistent winding
    if signed > 0:
        return poly[::-1]
    return poly


def bbox_to_band(
    bbox_xyxy: tuple[float, float, float, float],
    contact_mode: str,
    bottom_band_ratio: float,
) -> np.ndarray:
    x1, y1, x2, y2 = bbox_xyxy
    if contact_mode == "bottom_band":
        band_h = (y2 - y1) * bottom_band_ratio
        y1 = y2 - band_h
    return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=float)


def overlap_ratio(
    bbox_xyxy: tuple[float, float, float, float],
    wall_polygon: list[list[float]],
    contact_mode: str = "bottom_band",
    bottom_band_ratio: float = 0.25,
) -> float:
    """Fraction of the bbox (or its bottom band) area that lies inside the wall polygon.

    Returns 0.0 when there is no overlap. The denominator is the band area, so a
    bbox whose bottom band sits fully inside the wall returns ~1.0.
    """
    band = bbox_to_band(bbox_xyxy, contact_mode, bottom_band_ratio)
    band_area = _polygon_area(band)
    if band_area <= 0:
        return 0.0

    wall = np.array(wall_polygon, dtype=float)
    if len(wall) < 3:
        return 0.0
    wall = _ensure_ccw(wall)

    clipped = _clip_polygon(wall, band)
    if len(clipped) < 3:
        return 0.0

    inter_area = _polygon_area(clipped)
    return float(inter_area / band_area)


def touches_wall(
    bbox_xyxy: tuple[float, float, float, float],
    wall_polygon: list[list[float]],
    min_overlap_ratio: float,
    contact_mode: str = "bottom_band",
    bottom_band_ratio: float = 0.25,
) -> tuple[bool, float]:
    ratio = overlap_ratio(bbox_xyxy, wall_polygon, contact_mode, bottom_band_ratio)
    return ratio >= min_overlap_ratio, ratio
