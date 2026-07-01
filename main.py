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
    YOLO26_PT_PATH,
    YOLO26_RKNN_PATH,
)

logger = logging.getLogger("wallcrossing")


def mask_url_secret(url: str) -> str:
    return re.sub(r"://([^:/@]+):([^@]+)@", r"://\1:***@", str(url))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Wall-crossing detection service")
    parser.add_argument("--log-level", default="INFO", help="DEBUG, INFO, WARNING, ERROR")
    return parser.parse_args()


def log_startup_banner(cfg) -> None:
    enabled = cfg.enabled_cameras()
    logger.info("=" * 60)
    logger.info("Wall Crossing Service starting")
    logger.info("Service dir: %s", SERVICE_DIR)
    logger.info("Backend: %s", MODEL_BACKEND)
    logger.info("YOLO26 PT: %s", YOLO26_PT_PATH)
    logger.info("YOLO26 RKNN: %s", YOLO26_RKNN_PATH)
    logger.info("Image size: %s | confidence: %.2f", IMG_SIZE, DET_CONF_THRES)
    logger.info("Decode backend: %s", DECODE_BACKEND)
    logger.info("Enabled cameras: %d / %d", len(enabled), len(cfg.cameras))
    logger.info("Evidence dir: %s", EVIDENCE_DIR)
    logger.info("Alert log: %s", ALERT_LOG_PATH)
    logger.info("Log file: %s", LOG_FILE_PATH)
    for cam in enabled:
        logger.info("Camera %-18s %s", cam.id, mask_url_secret(cam.rtsp_url))
    logger.info("=" * 60)


def main() -> int:
    args = parse_args()
    setup_logging(args.log_level, LOG_FILE_PATH)

    cfg = load_runtime_config()
    log_startup_banner(cfg)

    pipeline = Pipeline(cfg)

    def request_stop(signum, frame):
        del frame
        logger.info("Received signal %s; stopping service...", signum)
        pipeline.stop()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    try:
        pipeline.run()
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received; stopping service...")
        pipeline.stop()
    finally:
        logger.info("Wall Crossing Service stopped")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(1)
