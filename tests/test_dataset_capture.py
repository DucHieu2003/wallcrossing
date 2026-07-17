import json

import cv2
import numpy as np

from wallcrossing.core.models import Detection
from wallcrossing.runtime.dataset_capture import DatasetCapture


def _label_path(root, image_path):
    relative = image_path.relative_to(root / "images")
    return root / "labels" / relative.with_suffix(".txt")


def test_capture_saves_raw_detection_yolo_label_and_metadata(tmp_path):
    capture = DatasetCapture(tmp_path, 5.0, 3600.0, 91)
    frame = np.full((12, 20, 3), 127, dtype=np.uint8)
    detection = Detection((1.0, 2.0, 10.0, 11.0), 0.87654, 0)

    out_path = capture.capture(
        camera_id="cam_001",
        timestamp="2026-07-14T03:14:58.194Z",
        now_mono=10.0,
        frame=frame,
        detections=[detection],
        overlap_ratios=[0.04321],
    )

    assert out_path is not None
    assert out_path.exists()
    assert "images/detections/2026-07-14/cam_001" in out_path.as_posix()
    saved = cv2.imread(str(out_path))
    assert saved.shape == frame.shape

    label_path = _label_path(tmp_path, out_path)
    assert label_path.read_text() == "0 0.275000 0.541667 0.450000 0.750000\n"

    record = json.loads((tmp_path / "metadata.jsonl").read_text())
    assert record["category"] == "detections"
    assert record["label_path"] == str(label_path)
    assert record["frame_width"] == 20
    assert record["frame_height"] == 12
    assert record["detections"][0] == {
        "bbox_xyxy": [1.0, 2.0, 10.0, 11.0],
        "confidence": 0.8765,
        "class_id": 0,
        "wall_overlap_ratio": 0.0432,
    }


def test_background_and_hard_negative_save_empty_yolo_labels(tmp_path):
    capture = DatasetCapture(tmp_path, 5.0, 60.0, 90, hard_negative_interval_seconds=120.0)
    frame = np.zeros((8, 8, 3), dtype=np.uint8)

    background = capture.capture(
        "cam_001", "2026-07-14T00:00:00.000Z", 10.0, frame, [], []
    )
    hard_negative = capture.capture(
        "cam_001",
        "2026-07-14T00:00:01.000Z",
        11.0,
        frame,
        [],
        [],
        hard_negative=True,
    )

    assert "images/background/" in background.as_posix()
    assert "images/hard_negatives/" in hard_negative.as_posix()
    assert _label_path(tmp_path, background).read_text() == ""
    assert _label_path(tmp_path, hard_negative).read_text() == ""


def test_yolo_labels_clip_boxes_to_frame_and_skip_invalid_boxes(tmp_path):
    capture = DatasetCapture(tmp_path, 0.0, 60.0, 90)
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    detections = [
        Detection((-20, -10, 220, 110), 0.9, 0),
        Detection((50, 50, 40, 60), 0.8, 0),
    ]

    image_path = capture.capture(
        "cam_001", "2026-07-14T00:00:00.000Z", 1.0, frame, detections, [0.0, 0.0]
    )

    assert _label_path(tmp_path, image_path).read_text() == (
        "0 0.500000 0.500000 1.000000 1.000000\n"
    )


def test_disk_cap_evicts_oldest_image_and_label_pair(tmp_path):
    rng = np.random.default_rng(0)
    capture = DatasetCapture(tmp_path, 0.0, 0.0, 90, max_disk_gb=1.0)
    detection = Detection((1, 1, 10, 10), 0.9, 0)

    def noise():
        return rng.integers(0, 256, size=(120, 120, 3), dtype=np.uint8)

    first = capture.capture(
        "cam_001", "2026-07-15T00:00:00.000Z", 0.0, noise(), [detection], [0.2]
    )
    first_label = _label_path(tmp_path, first)
    capture.max_disk_bytes = int(capture._tracked_bytes * 2)

    paths = [first]
    for i in range(1, 8):
        paths.append(
            capture.capture(
                "cam_001",
                f"2026-07-15T00:00:0{i}.000Z",
                float(i),
                noise(),
                [detection],
                [0.2],
            )
        )

    assert capture._tracked_bytes <= capture.max_disk_bytes
    assert not paths[0].exists()
    assert not first_label.exists()
    assert paths[-1].exists()
    assert _label_path(tmp_path, paths[-1]).exists()


def test_capture_rate_limits_categories_independently(tmp_path):
    capture = DatasetCapture(tmp_path, 5.0, 60.0, 90, hard_negative_interval_seconds=120.0)
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    detection = Detection((1, 1, 4, 7), 0.9, 0)

    first_detection = capture.capture(
        "cam_001", "2026-07-14T00:00:00.000Z", 10.0, frame, [detection], [0.2]
    )
    skipped_detection = capture.capture(
        "cam_001", "2026-07-14T00:00:01.000Z", 11.0, frame, [detection], [0.2]
    )
    first_background = capture.capture(
        "cam_001", "2026-07-14T00:00:02.000Z", 12.0, frame, [], []
    )
    first_hard_negative = capture.capture(
        "cam_001",
        "2026-07-14T00:00:03.000Z",
        13.0,
        frame,
        [],
        [],
        hard_negative=True,
    )
    next_detection = capture.capture(
        "cam_001", "2026-07-14T00:00:15.000Z", 15.0, frame, [detection], [0.2]
    )

    assert first_detection is not None
    assert skipped_detection is None
    assert first_background is not None
    assert first_hard_negative is not None
    assert next_detection is not None
    assert len((tmp_path / "metadata.jsonl").read_text().splitlines()) == 4
