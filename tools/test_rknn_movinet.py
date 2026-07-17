"""Run RKNN MoViNet shovel-behavior model on a video clip.

Example:
  python3 tools/test_rknn_movinet.py \
    --model wallcrossing/weights/movinet_scoop_rk3588_fp16.rknn \
    --video sample.mp4

Model contract:
- input:  float32 [1, 50, 172, 172, 3], RGB, pixel range [0, 1]
- output: logits [1, 2]
- class order: normal, steal
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np


CLASSES = ("normal", "steal")


def softmax(logits: np.ndarray) -> np.ndarray:
    x = logits.astype(np.float32).reshape(-1)
    x = x - np.max(x)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x)


def read_clip(
    video_path: str,
    num_frames: int = 50,
    image_size: int = 172,
) -> np.ndarray:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"failed to open video: {video_path}")

    frames: list[np.ndarray] = []
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if total > 0:
            indices = np.linspace(0, max(total - 1, 0), num_frames).astype(int).tolist()
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ok, frame = cap.read()
                if not ok:
                    break
                frames.append(preprocess_frame(frame, image_size))
        else:
            while len(frames) < num_frames:
                ok, frame = cap.read()
                if not ok:
                    break
                frames.append(preprocess_frame(frame, image_size))
    finally:
        cap.release()

    if not frames:
        raise RuntimeError(f"no frames decoded: {video_path}")

    while len(frames) < num_frames:
        frames.append(frames[-1].copy())

    clip = np.stack(frames[:num_frames], axis=0).astype(np.float32)
    return np.expand_dims(clip, axis=0)


def preprocess_frame(frame_bgr: np.ndarray, image_size: int) -> np.ndarray:
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    frame_rgb = center_crop_square(frame_rgb)
    frame_rgb = cv2.resize(frame_rgb, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    return frame_rgb.astype(np.float32) / 255.0


def center_crop_square(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    side = min(height, width)
    y0 = (height - side) // 2
    x0 = (width - side) // 2
    return image[y0 : y0 + side, x0 : x0 + side]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Test RKNN MoViNet action model")
    parser.add_argument("--model", default="wallcrossing/weights/movinet_scoop_rk3588_fp16.rknn")
    parser.add_argument("--video", required=True)
    parser.add_argument("--frames", type=int, default=50)
    parser.add_argument("--size", type=int, default=172)
    parser.add_argument(
        "--target",
        help="optional RKNN target, for example rk3588; omit for local NPU runtime",
    )
    args = parser.parse_args(argv)

    inp = read_clip(args.video, num_frames=args.frames, image_size=args.size)
    if inp.shape != (1, 50, 172, 172, 3):
        raise ValueError(f"unexpected input shape: {inp.shape}")

    from rknn.api import RKNN

    rknn = RKNN(verbose=False)
    try:
        if rknn.load_rknn(args.model) != 0:
            raise RuntimeError(f"load_rknn failed: {args.model}")

        if args.target:
            ret = rknn.init_runtime(target=args.target)
        else:
            ret = rknn.init_runtime()
        if ret != 0:
            raise RuntimeError("init_runtime failed")

        outputs = rknn.inference(inputs=[inp])
    finally:
        rknn.release()

    if not outputs:
        raise RuntimeError("empty RKNN outputs")

    logits = np.asarray(outputs[0], dtype=np.float32).reshape(-1)
    probs = softmax(logits)
    pred_idx = int(np.argmax(probs))

    print(f"logits: {logits.tolist()}")
    print(f"normal_prob: {probs[0]:.6f}")
    print(f"steal_prob:  {probs[1]:.6f}")
    print(f"predict: {CLASSES[pred_idx]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
