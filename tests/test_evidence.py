import cv2
import numpy as np

from wallcrossing.alerts.evidence import (
    debug_preview_path,
    draw_detections_preview,
)
from wallcrossing.core.models import Detection


def test_debug_preview_path_uses_date_and_camera_dirs(tmp_path):
    path = debug_preview_path(tmp_path, "cam_001", "2026-07-15T03:14:58.194Z")

    assert "2026-07-15/cam_001" in path.as_posix()
    assert path.suffix == ".jpg"


def test_draw_detections_preview_saves_all_bboxes(tmp_path):
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    polygon = [[10, 10], [50, 10], [50, 90], [10, 90]]
    detections = [
        Detection((20, 20, 40, 60), 0.91, 0),
        Detection((120, 30, 160, 80), 0.55, 0),
    ]
    out_path = tmp_path / "preview.jpg"

    draw_detections_preview(frame, polygon, detections, "cam n=2", out_path)

    assert out_path.exists()
    saved = cv2.imread(str(out_path))
    assert saved.shape == frame.shape
    # boxes/polygon are drawn, so the saved frame is no longer all-black
    assert saved.sum() > 0
