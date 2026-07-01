from __future__ import annotations

import cv2
import numpy as np

from wallcrossing.core.models import Detection


def letterbox(image: np.ndarray, imgsz: int) -> np.ndarray:
    """Resize keeping aspect ratio, pad to a square imgsz x imgsz, return RGB uint8 NHWC.

    The padding offsets/scale are recomputed in decode_yolo from orig_shape, so we
    don't return them here — keep this matched with _unletterbox below.
    """
    h, w = image.shape[:2]
    scale = min(imgsz / h, imgsz / w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((imgsz, imgsz, 3), 114, dtype=np.uint8)
    top = (imgsz - nh) // 2
    left = (imgsz - nw) // 2
    canvas[top : top + nh, left : left + nw] = resized
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    return rgb[np.newaxis, ...]


def _unletterbox_box(
    box: np.ndarray, orig_shape: tuple[int, int], imgsz: int
) -> np.ndarray:
    h, w = orig_shape
    scale = min(imgsz / h, imgsz / w)
    nh, nw = h * scale, w * scale
    top = (imgsz - nh) / 2
    left = (imgsz - nw) / 2
    box = box.copy()
    box[[0, 2]] = (box[[0, 2]] - left) / scale
    box[[1, 3]] = (box[[1, 3]] - top) / scale
    box[[0, 2]] = box[[0, 2]].clip(0, w)
    box[[1, 3]] = box[[1, 3]].clip(0, h)
    return box


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_thres: float = 0.45) -> list[int]:
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[1:][iou <= iou_thres]
    return keep


def decode_yolo(
    outputs: list[np.ndarray],
    orig_shape: tuple[int, int],
    imgsz: int,
    conf_thres: float,
    person_class_id: int,
    iou_thres: float = 0.45,
) -> list[Detection]:
    """Decode raw YOLO output into person Detections in original-image coordinates.

    Assumes a single output tensor shaped (1, 4+num_classes, num_anchors) — the
    common Ultralytics export layout. Boxes are xywh (center) in letterboxed space.
    Adjust here if the exported .rknn uses a different head layout.
    """
    pred = outputs[0]
    pred = np.squeeze(pred, axis=0)  # (4+nc, N)
    if pred.shape[0] < pred.shape[1]:
        pred = pred.transpose()  # -> (N, 4+nc)

    boxes_xywh = pred[:, :4]
    cls_scores = pred[:, 4:]

    cls_id = cls_scores.argmax(axis=1)
    cls_conf = cls_scores.max(axis=1)

    mask = (cls_id == person_class_id) & (cls_conf >= conf_thres)
    if not mask.any():
        return []

    boxes_xywh = boxes_xywh[mask]
    cls_conf = cls_conf[mask]

    xyxy = np.empty_like(boxes_xywh)
    xyxy[:, 0] = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2
    xyxy[:, 1] = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2
    xyxy[:, 2] = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2
    xyxy[:, 3] = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2

    keep = _nms(xyxy, cls_conf, iou_thres)

    detections: list[Detection] = []
    for i in keep:
        box = _unletterbox_box(xyxy[i], orig_shape, imgsz)
        detections.append(
            Detection(
                bbox_xyxy=(float(box[0]), float(box[1]), float(box[2]), float(box[3])),
                confidence=float(cls_conf[i]),
                class_id=person_class_id,
            )
        )
    return detections
