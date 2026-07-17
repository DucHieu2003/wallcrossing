from __future__ import annotations

import logging
from typing import Protocol

import numpy as np

from wallcrossing.core.config_loader import ModelConfig
from wallcrossing.core.models import Detection

logger = logging.getLogger("wallcrossing.detector")


class Detector(Protocol):
    def detect(self, image: np.ndarray) -> list[Detection]: ...

    def close(self) -> None: ...


class MockDetector:
    """Tra ve detection dung san. Cho test va chay thu pipeline khong can model."""

    def __init__(self, person_class_id: int = 0, scripted: list[Detection] | None = None):
        self.person_class_id = person_class_id
        self._scripted = scripted or []

    def detect(self, image: np.ndarray) -> list[Detection]:
        return list(self._scripted)

    def close(self) -> None:
        pass


class RknnDetector:
    """Backend tren box: YOLO26s FP16 .rknn qua rknn-toolkit2 (RKNN)."""

    def __init__(self, cfg: ModelConfig):
        from rknn.api import RKNN

        self.cfg = cfg
        self.rknn = RKNN()
        if self.rknn.load_rknn(cfg.rknn_path) != 0:
            raise RuntimeError(f"khong tai duoc model rknn: {cfg.rknn_path}")
        core_mask = self._core_mask(cfg.npu_cores)
        if self.rknn.init_runtime(target="rk3588", core_mask=core_mask) != 0:
            raise RuntimeError("khong khoi tao duoc runtime rknn")

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
        outputs = self.rknn.inference(inputs=[inp], data_format="nhwc")
        return postprocess.decode_yolo(
            outputs,
            orig_shape=image.shape[:2],
            imgsz=self.cfg.imgsz,
            conf_thres=self.cfg.confidence,
            person_class_id=self.cfg.person_class_id,
        )

    def close(self) -> None:
        if self.rknn is not None:
            self.rknn.release()
            self.rknn = None


def build_detector(cfg: ModelConfig) -> Detector:
    if cfg.backend == "mock":
        return MockDetector(person_class_id=cfg.person_class_id)
    if cfg.backend == "rknn":
        logger.info("dung backend rknn, model=%s cores=%s", cfg.rknn_path, cfg.npu_cores)
        return RknnDetector(cfg)
    raise ValueError(f"backend khong hop le: {cfg.backend}")
