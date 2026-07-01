"""Convert YOLO26s .pt -> ONNX -> .rknn (FP16) for RK3588.

Run this on an x86 host with rknn-toolkit2 installed (NOT on the box).
The box only needs the resulting .rknn + rknn-toolkit-lite2.

  python tools/convert_rknn.py --pt weights/yolo26s.pt --out weights/yolo26s_rk3588_fp16.rknn

Notes:
- We export the model WITHOUT the head decode block so the .rknn outputs raw
  tensors; decode + NMS run on CPU in wallcrossing/postprocess.py. This avoids
  unsupported ops on the NPU. If your export already includes decode, adjust
  postprocess.decode_yolo accordingly.
- FP16: do_quantization=False, so no calibration dataset is needed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def export_onnx(pt_path: str, imgsz: int) -> str:
    from ultralytics import YOLO

    model = YOLO(pt_path)
    onnx_path = model.export(format="onnx", imgsz=imgsz, opset=12, simplify=True)
    return str(onnx_path)


def convert_rknn(onnx_path: str, out_path: str, imgsz: int) -> None:
    from rknn.api import RKNN

    rknn = RKNN(verbose=True)
    rknn.config(
        mean_values=[[0, 0, 0]],
        std_values=[[255, 255, 255]],
        target_platform="rk3588",
    )
    if rknn.load_onnx(model=onnx_path) != 0:
        raise RuntimeError("load_onnx failed")
    # FP16: no quantization, no calibration dataset
    if rknn.build(do_quantization=False) != 0:
        raise RuntimeError("build failed")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    if rknn.export_rknn(out_path) != 0:
        raise RuntimeError("export_rknn failed")
    rknn.release()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Convert YOLO .pt to RK3588 .rknn (FP16)")
    parser.add_argument("--pt", required=True, help="path to YOLO .pt")
    parser.add_argument("--out", required=True, help="output .rknn path")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--onnx", help="reuse an existing .onnx instead of re-exporting")
    args = parser.parse_args(argv)

    onnx_path = args.onnx or export_onnx(args.pt, args.imgsz)
    print(f"onnx: {onnx_path}")
    convert_rknn(onnx_path, args.out, args.imgsz)
    print(f"rknn: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
