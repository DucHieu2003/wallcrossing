"""Run RKNN YOLO model on one image and keep only class person.

Example:
  python tools/test_rknn_person.py \
    --model wallcrossing/weights/yolo26s_rk3588_fp16.rknn \
    --image sample.jpg \
    --out outputs/test_person.jpg
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from wallcrossing.detection.postprocess import decode_yolo, letterbox  # noqa: E402


def draw_persons(image, detections, color=(0, 255, 0)):
    out = image.copy()
    for det in detections:
        x1, y1, x2, y2 = map(int, det.bbox_xyxy)
        label = f"person {det.confidence:.2f}"
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            out,
            label,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Test RKNN model, person class only")
    parser.add_argument("--model", default="wallcrossing/weights/yolo26s_rk3588_fp16.rknn")
    parser.add_argument("--image", required=True)
    parser.add_argument("--out", default="outputs/test_person.jpg")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--person-class-id", type=int, default=0)
    parser.add_argument(
        "--target",
        help="optional RKNN target, for example rk3588; omit for local NPU runtime",
    )
    args = parser.parse_args(argv)

    image = cv2.imread(args.image)
    if image is None:
        raise FileNotFoundError(f"failed to read image: {args.image}")

    from rknn.api import RKNN

    rknn = RKNN(verbose=False)
    try:
        if rknn.load_rknn(args.model) != 0:
            raise RuntimeError(f"load_rknn failed: {args.model}")

        # On RK3588 board, omit --target to use local NPU runtime.
        if args.target:
            ret = rknn.init_runtime(target=args.target)
        else:
            ret = rknn.init_runtime()
        if ret != 0:
            raise RuntimeError("init_runtime failed")

        inp = letterbox(image, args.imgsz)
        outputs = rknn.inference(inputs=[inp])
        detections = decode_yolo(
            outputs,
            orig_shape=image.shape[:2],
            imgsz=args.imgsz,
            conf_thres=args.conf,
            person_class_id=args.person_class_id,
            iou_thres=args.iou,
        )
    finally:
        rknn.release()

    print(f"persons: {len(detections)}")
    for i, det in enumerate(detections, start=1):
        x1, y1, x2, y2 = det.bbox_xyxy
        print(
            f"{i}: conf={det.confidence:.4f} "
            f"bbox=({x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f})"
        )

    out_image = draw_persons(image, detections)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), out_image)
    print(f"saved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
