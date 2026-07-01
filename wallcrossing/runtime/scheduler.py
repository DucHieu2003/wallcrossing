from __future__ import annotations


class FrameScheduler:
    """Decides when each camera is next due for a detection based on its detect_fps.

    detect_fps == 0 means "as fast as possible" (always due). Otherwise a camera
    is due once 1/detect_fps seconds have passed since its last detection.
    """

    def __init__(self, detect_fps: dict[str, float]):
        self._interval = {
            cam_id: (1.0 / fps if fps > 0 else 0.0) for cam_id, fps in detect_fps.items()
        }
        self._next_due: dict[str, float] = {cam_id: 0.0 for cam_id in detect_fps}

    def due_cameras(self, now_mono: float) -> list[str]:
        return [cam_id for cam_id, due in self._next_due.items() if now_mono >= due]

    def mark_done(self, cam_id: str, now_mono: float) -> None:
        self._next_due[cam_id] = now_mono + self._interval[cam_id]
