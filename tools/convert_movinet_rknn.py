"""Convert MoViNet shovel-behavior TFLite -> RKNN FP16 for RK3588.

Run this on a convert host with rknn-toolkit2 installed.
The RK3588 box only needs the resulting .rknn + rknn-toolkit-lite2.

  python3 tools/convert_movinet_rknn.py \
    --tflite wallcrossing/weights/movinet_scoop_fp32.tflite \
    --out wallcrossing/weights/movinet_scoop_rk3588_fp16.rknn

Model contract from export log:
- input:  float32 [1, 50, 172, 172, 3], RGB, pixel range [0, 1]
- output: logits float32 [1, 2]
- class order: normal, steal
- softmax is outside the model

Notes:
- No mean/std is configured here. Feed RKNN a float32 video tensor already
  normalized to RGB [0, 1]. RKNN image pre-process knobs are safer for 4D
  image models than for this 5D video input.
- FP16 build uses do_quantization=False, so no calibration dataset is needed.
- If RKNN Toolkit fails on input rank 5 or Conv3D/3D video ops, this model is
  not directly supported by current RKNN convert path and needs model rewrite.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


DEFAULT_TFLITE = "wallcrossing/weights/movinet_scoop_fp32.tflite"
DEFAULT_OUT = "wallcrossing/weights/movinet_scoop_rk3588_fp16.rknn"


class ConversionError(RuntimeError):
    """RKNN conversion step failed."""


def convert_movinet_tflite_to_rknn(
    tflite_path: str,
    out_path: str,
    target_platform: str = "rk3588",
    float_dtype: str = "float16",
    verbose: bool = True,
) -> None:
    from rknn.api import RKNN

    model_path = Path(tflite_path)
    if not model_path.is_file():
        raise FileNotFoundError(f"TFLite model not found: {model_path}")

    rknn = RKNN(verbose=verbose)
    try:
        ret = rknn.config(
            target_platform=target_platform,
            float_dtype=float_dtype,
        )
        if ret != 0:
            raise ConversionError("rknn.config failed")

        # TFLite export is channels-last: [N, T, H, W, C].
        # input_is_nchw must stay False.
        ret = rknn.load_tflite(model=str(model_path), input_is_nchw=False)
        if ret != 0:
            raise ConversionError(
                "rknn.load_tflite failed. If log mentions 5D input or Conv3D, "
                "RKNN Toolkit does not support this MoViNet graph directly."
            )

        # FP16, no calibration dataset, keep model input as float tensor.
        ret = rknn.build(do_quantization=False)
        if ret != 0:
            raise ConversionError(
                "rknn.build failed. Check unsupported-op lines in RKNN log."
            )

        output_path = Path(out_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        ret = rknn.export_rknn(str(output_path))
        if ret != 0:
            raise ConversionError("rknn.export_rknn failed")
    finally:
        rknn.release()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Convert MoViNet shovel-behavior TFLite to RK3588 RKNN FP16"
    )
    parser.add_argument("--tflite", default=DEFAULT_TFLITE, help="input .tflite path")
    parser.add_argument("--out", default=DEFAULT_OUT, help="output .rknn path")
    parser.add_argument("--target-platform", default="rk3588", help="RKNN target")
    parser.add_argument(
        "--float-dtype",
        default="float16",
        choices=("float16", "float32"),
        help="floating-point dtype used by RKNN build",
    )
    parser.add_argument("--quiet", action="store_true", help="disable RKNN verbose logs")
    args = parser.parse_args(argv)

    convert_movinet_tflite_to_rknn(
        tflite_path=args.tflite,
        out_path=args.out,
        target_platform=args.target_platform,
        float_dtype=args.float_dtype,
        verbose=not args.quiet,
    )

    print(f"tflite: {args.tflite}")
    print(f"rknn:   {args.out}")
    print("input:  float32 [1,50,172,172,3] RGB [0,1]")
    print("output: logits [1,2]; apply softmax outside; classes: normal, steal")
    return 0


if __name__ == "__main__":
    sys.exit(main())
