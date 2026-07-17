import numpy as np

from wallcrossing.core.models import Detection
from wallcrossing.runtime.motion_filter import MotionFilter


def _det(x1, y1, x2, y2):
    return Detection((x1, y1, x2, y2), 0.9, 0)


def test_static_detection_dropped_moving_detection_kept():
    mf = MotionFilter(min_motion_ratio=0.02, warmup_frames=3)
    cam = "cam_001"
    background = np.full((240, 320, 3), 100, dtype=np.uint8)

    # build the background model past warmup with steady frames
    for _ in range(6):
        mf.filter(cam, background, [], [])

    # a person-sized region on the right changes; the left stays static (tarp)
    frame = background.copy()
    frame[40:200, 220:300] = 240

    static_det = _det(10, 40, 90, 200)
    moving_det = _det(220, 40, 300, 200)

    kept, overlaps = mf.filter(
        cam, frame, [static_det, moving_det], [0.1, 0.2]
    )

    assert moving_det in kept
    assert static_det not in kept
    assert overlaps == [0.2]


def test_returns_empty_during_warmup():
    mf = MotionFilter(warmup_frames=10)
    frame = np.full((120, 160, 3), 50, dtype=np.uint8)
    kept, overlaps = mf.filter("cam_001", frame, [_det(1, 1, 20, 20)], [0.3])
    assert kept == []
    assert overlaps == []
