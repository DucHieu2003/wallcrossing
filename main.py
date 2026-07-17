from __future__ import annotations

import argparse
import logging
import re
import signal
import sys

from wallcrossing.core.config_loader import load_runtime_config
from wallcrossing.runtime.pipeline import Pipeline
from wallcrossing.utils.logging_setup import setup_logging

from config import (
    ALERT_LOG_PATH,
    DECODE_BACKEND,
    DET_CONF_THRES,
    EVIDENCE_DIR,
    IMG_SIZE,
    LOG_FILE_PATH,
    MODEL_BACKEND,
    SERVICE_DIR,
    YOLO26_RKNN_PATH,
)

logger = logging.getLogger("wallcrossing")


def mask_url_secret(url: str) -> str:
    return re.sub(r"://([^:/@]+):([^@]+)@", r"://\1:***@", str(url))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dich vu phat hien vuot tuong")
    parser.add_argument("--log-level", default="INFO", help="DEBUG, INFO, WARNING, ERROR")
    return parser.parse_args()


def log_startup_banner(cfg) -> None:
    enabled = cfg.enabled_cameras()
    logger.info("=" * 60)
    logger.info("Dich vu phat hien vuot tuong dang khoi dong")
    logger.info("Thu muc dich vu: %s", SERVICE_DIR)
    logger.info("Backend: %s", MODEL_BACKEND)
    logger.info("YOLO26 RKNN: %s", YOLO26_RKNN_PATH)
    logger.info("Kich thuoc anh: %s | nguong tin cay: %.2f", IMG_SIZE, DET_CONF_THRES)
    logger.info("Backend giai ma: %s", DECODE_BACKEND)
    logger.info("Camera bat: %d / %d", len(enabled), len(cfg.cameras))
    logger.info("Che do detect: %s", "ROI" if cfg.pipeline.detect_roi_enabled else "toan khung hinh")
    logger.info(
        "Thu thap dataset: bat=%s thu_muc=%s gioi_han=%.1fGB loc_chuyen_dong=%s",
        cfg.pipeline.dataset_capture_enabled,
        cfg.pipeline.dataset_capture_dir,
        cfg.pipeline.dataset_max_disk_gb,
        cfg.pipeline.motion_filter_enabled,
    )
    logger.info(
        "Canh bao: bat=%s evidence=%s gioi_han=%.1fGB log=%s xoay=%dMBx%d",
        cfg.pipeline.alerts_enabled,
        EVIDENCE_DIR,
        cfg.pipeline.evidence_max_disk_gb,
        ALERT_LOG_PATH,
        cfg.pipeline.alert_log_max_mb,
        cfg.pipeline.alert_log_backup_count + 1,
    )
    logger.info(
        "Bao ve bo nho: nguong_restart_rss=%.0fMB",
        cfg.pipeline.rss_graceful_restart_mb,
    )
    logger.info("File log: %s", LOG_FILE_PATH)
    for cam in enabled:
        logger.info("Camera %-18s %s", cam.id, mask_url_secret(cam.rtsp_url))
    logger.info("=" * 60)


def quiet_opencv_warnings() -> None:
    """Chi giu loi OpenCV; canh bao reconnect GStreamer spam lam nhieu log cua ta."""
    try:
        import cv2

        cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
    except Exception:
        pass


def main() -> int:
    args = parse_args()
    setup_logging(args.log_level, LOG_FILE_PATH)
    quiet_opencv_warnings()

    cfg = load_runtime_config()
    log_startup_banner(cfg)

    pipeline = Pipeline(cfg)

    def request_stop(signum, frame):
        del frame
        logger.info("Nhan tin hieu %s; yeu cau dung dich vu...", signum)
        pipeline.request_stop()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    try:
        pipeline.run()
    except KeyboardInterrupt:
        logger.info("Nhan KeyboardInterrupt; dang dung dich vu...")
        pipeline.stop()
    finally:
        logger.info("Dich vu phat hien vuot tuong da dung")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"LOI: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(1)
