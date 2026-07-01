import numpy as np

from wallcrossing.detection.postprocess import decode_yolo


def test_decode_end_to_end_output_filters_person_class():
    out = np.zeros((1, 300, 6), dtype=np.float32)
    out[0, 0] = [100, 120, 200, 320, 0.9, 0]
    out[0, 1] = [10, 20, 30, 40, 0.95, 1]
    out[0, 2] = [50, 60, 70, 80, 0.1, 0]

    detections = decode_yolo(
        [out],
        orig_shape=(640, 640),
        imgsz=640,
        conf_thres=0.45,
        person_class_id=0,
    )

    assert len(detections) == 1
    assert detections[0].bbox_xyxy == (100.0, 120.0, 200.0, 320.0)
    assert detections[0].confidence == 0.8999999761581421
    assert detections[0].class_id == 0


def test_decode_raw_head_output_still_supported():
    out = np.zeros((1, 84, 1), dtype=np.float32)
    out[0, 0:4, 0] = [150, 220, 100, 200]
    out[0, 4, 0] = 0.9

    detections = decode_yolo(
        [out],
        orig_shape=(640, 640),
        imgsz=640,
        conf_thres=0.45,
        person_class_id=0,
    )

    assert len(detections) == 1
    assert detections[0].bbox_xyxy == (100.0, 120.0, 200.0, 320.0)
