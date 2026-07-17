from __future__ import annotations

import cv2
import numpy as np

from wallcrossing.core.models import Detection


class MotionFilter:
    """Keep only detections whose bbox overlaps recent motion, per camera.

    Static false positives (a tarp, a painted mark on the wall) never move, so
    once the per-camera background model settles their bbox has almost no
    foreground pixels. A real person moves and lights up the foreground mask.
    Used to stop the fine-tuning dataset from filling up with the same tarp.
    """

    def __init__(
        self,
        min_motion_ratio: float = 0.05,
        downscale_width: int = 320,
        history: int = 500,
        var_threshold: float = 16.0,
        warmup_frames: int = 30,
    ):
        self.min_motion_ratio = min_motion_ratio
        self.downscale_width = downscale_width
        self.history = history
        self.var_threshold = var_threshold
        self.warmup_frames = warmup_frames
        self._subtractors: dict[str, cv2.BackgroundSubtractor] = {}
        self._seen: dict[str, int] = {}
        self._kernel = np.ones((3, 3), np.uint8)

    def _mask_for(self, camera_id: str, frame: np.ndarray) -> tuple[np.ndarray, float]:
        h, w = frame.shape[:2]
        scale = self.downscale_width / w if w > self.downscale_width else 1.0
        if scale < 1.0:
            small = cv2.resize(
                frame, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA
            )
        else:
            small = frame
        sub = self._subtractors.get(camera_id)
        if sub is None:
            sub = cv2.createBackgroundSubtractorMOG2(
                history=self.history, varThreshold=self.var_threshold, detectShadows=False
            )
            self._subtractors[camera_id] = sub
        mask = sub.apply(small)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._kernel)
        return mask, scale

    def is_ready(self, camera_id: str) -> bool:
        return self._seen.get(camera_id, 0) >= self.warmup_frames

    def filter(
        self,
        camera_id: str,
        frame: np.ndarray,
        detections: list[Detection],
        overlap_ratios: list[float],
    ) -> tuple[list[Detection], list[float]]:
        # Always feed the frame so the background model stays current, even on
        # frames with no detections.
        mask, scale = self._mask_for(camera_id, frame)
        seen = self._seen.get(camera_id, 0) + 1
        self._seen[camera_id] = seen
        if not detections or seen < self.warmup_frames:
            return [], []

        mh, mw = mask.shape[:2]
        kept: list[Detection] = []
        kept_overlaps: list[float] = []
        for det, overlap in zip(detections, overlap_ratios):
            x1, y1, x2, y2 = det.bbox_xyxy
            mx1 = max(0, min(mw - 1, int(x1 * scale)))
            my1 = max(0, min(mh - 1, int(y1 * scale)))
            mx2 = max(mx1 + 1, min(mw, int(x2 * scale)))
            my2 = max(my1 + 1, min(mh, int(y2 * scale)))
            roi = mask[my1:my2, mx1:mx2]
            if roi.size == 0:
                continue
            motion_ratio = float(np.count_nonzero(roi)) / roi.size
            if motion_ratio >= self.min_motion_ratio:
                kept.append(det)
                kept_overlaps.append(overlap)
        return kept, kept_overlaps
