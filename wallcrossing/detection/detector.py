from __future__ import annotations

import logging
from typing import Protocol

import numpy as np

from wallcrossing.core.config_loader import ModelConfig
from wallcrossing.core.models import Detection

logger = logging.getLogger("wallcrossing.detector")


class Detector(Protocol):
    def detect(self, image: np.ndarray) -> list[Detection]: ...


class MockDetector:
    """Returns pre-seeded detections. For tests and pipeline dry-runs without a model."""

    def __init__(self, person_class_id: int = 0, scripted: list[Detection] | None = None):
        self.person_class_id = person_class_id
        self._scripted = scripted or []

    def detect(self, image: np.ndarray) -> list[Detection]:
        return list(self._scripted)


class UltralyticsDetector:
    """Host-dev backend using the original .pt via ultralytics. Not used on the box."""

    def __init__(self, cfg: ModelConfig):
        from ultralytics import YOLO

        self.cfg = cfg
        self.model = YOLO(cfg.pt_path)

    def detect(self, image: np.ndarray) -> list[Detection]:
        results = self.model.predict(
            image,
            imgsz=self.cfg.imgsz,
            conf=self.cfg.confidence,
            classes=[self.cfg.person_class_id],
            verbose=False,
        )
        out: list[Detection] = []
        for r in results:
            for box in r.boxes:
                xyxy = box.xyxy[0].tolist()
                out.append(
                    Detection(
                        bbox_xyxy=(xyxy[0], xyxy[1], xyxy[2], xyxy[3]),
                        confidence=float(box.conf[0]),
                        class_id=int(box.cls[0]),
                    )
                )
        return out


class RknnDetector:
    """On-box backend: YOLO26s FP16 .rknn via rknn-toolkit2 (RKNN)."""

    def __init__(self, cfg: ModelConfig):
        from rknn.api import RKNN

        self.cfg = cfg
        self.rknn = RKNN()
        if self.rknn.load_rknn(cfg.rknn_path) != 0:
            raise RuntimeError(f"failed to load rknn model: {cfg.rknn_path}")
        core_mask = self._core_mask(cfg.npu_cores)
        if self.rknn.init_runtime(core_mask=core_mask) != 0:
            raise RuntimeError("failed to init rknn runtime")

    @staticmethod
    def _core_mask(cores: list[int]):
        from rknn.api import RKNN

        table = {
            0: RKNN.NPU_CORE_0,
            1: RKNN.NPU_CORE_1,
            2: RKNN.NPU_CORE_2,
        }
        if set(cores) >= {0, 1, 2}:
            return RKNN.NPU_CORE_0_1_2
        mask = 0
        for c in cores:
            mask |= table[c]
        return mask

    def detect(self, image: np.ndarray) -> list[Detection]:
        from wallcrossing.detection import postprocess

        inp = postprocess.letterbox(image, self.cfg.imgsz)
        outputs = self.rknn.inference(inputs=[inp])
        return postprocess.decode_yolo(
            outputs,
            orig_shape=image.shape[:2],
            imgsz=self.cfg.imgsz,
            conf_thres=self.cfg.confidence,
            person_class_id=self.cfg.person_class_id,
        )


def build_detector(cfg: ModelConfig) -> Detector:
    if cfg.backend == "mock":
        return MockDetector(person_class_id=cfg.person_class_id)
    if cfg.backend == "ultralytics":
        logger.info("using ultralytics backend (host dev), model=%s", cfg.pt_path)
        return UltralyticsDetector(cfg)
    if cfg.backend == "rknn":
        logger.info("using rknn backend, model=%s cores=%s", cfg.rknn_path, cfg.npu_cores)
        return RknnDetector(cfg)
    raise ValueError(f"unknown backend: {cfg.backend}")
