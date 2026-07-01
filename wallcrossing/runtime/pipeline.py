from __future__ import annotations

import logging
import time
from datetime import datetime, timezone

from wallcrossing.alerts.alerting import AlertManager
from wallcrossing.alerts.evidence import draw_and_save, evidence_path
from wallcrossing.core.config_loader import AppConfig
from wallcrossing.core.models import AlertEvent
from wallcrossing.detection.detector import build_detector
from wallcrossing.services.wall_contact import touches_wall
from wallcrossing.runtime.scheduler import FrameScheduler
from wallcrossing.streams.rtsp_reader import RtspReader

logger = logging.getLogger("wallcrossing.pipeline")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


class Pipeline:
    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        self.detector = build_detector(cfg.model)
        self.alerts = AlertManager(
            consecutive_hits=cfg.rules.consecutive_hits,
            cooldown_seconds=cfg.rules.cooldown_seconds,
            alert_log_path=cfg.pipeline.alert_log_path,
        )
        self.cameras = {c.id: c for c in cfg.enabled_cameras()}
        self.readers: dict[str, RtspReader] = {
            c.id: RtspReader(c.id, c.rtsp_url, decode_backend=cfg.pipeline.decode_backend)
            for c in self.cameras.values()
        }
        self.scheduler = FrameScheduler(
            {c.id: cfg.detect_fps_for(c) for c in self.cameras.values()}
        )
        self._stop = False

    def start(self) -> None:
        for r in self.readers.values():
            r.start()
        logger.info("started %d camera readers", len(self.readers))

    def stop(self) -> None:
        self._stop = True
        for r in self.readers.values():
            r.stop()

    def _process_camera(self, cam_id: str, now_mono: float) -> None:
        cam = self.cameras[cam_id]
        reader = self.readers[cam_id]
        frame, _ = reader.read_latest()
        if frame is None:
            return

        detections = self.detector.detect(frame)

        best = None
        best_ratio = 0.0
        for det in detections:
            hit, ratio = touches_wall(
                det.bbox_xyxy,
                cam.wall_polygon,
                self.cfg.rules.min_overlap_ratio,
                self.cfg.rules.contact_mode,
                self.cfg.rules.bottom_band_ratio,
            )
            if hit and ratio > best_ratio:
                best = det
                best_ratio = ratio

        should_fire = self.alerts.update(cam_id, best is not None, now_mono)
        if not should_fire or best is None:
            return

        ts = _utc_now_iso()
        alert_id = f"{cam_id}-{ts.replace(':', '').replace('-', '').replace('.', '')}"
        out_path = evidence_path(self.cfg.pipeline.evidence_dir, cam_id, ts, alert_id)
        label = f"{cam.name or cam_id} {best.confidence:.2f} {ts}"
        draw_and_save(frame, cam.wall_polygon, best, label, out_path)

        event = AlertEvent(
            alert_id=alert_id,
            camera_id=cam_id,
            camera_name=cam.name,
            timestamp=ts,
            bbox_xyxy=best.bbox_xyxy,
            confidence=best.confidence,
            wall_polygon=cam.wall_polygon,
            overlap_ratio=best_ratio,
            evidence_path=str(out_path),
        )
        self.alerts.write(event)

    def run(self) -> None:
        self.start()
        try:
            while not self._stop:
                now = time.monotonic()
                due = self.scheduler.due_cameras(now)
                if not due:
                    time.sleep(0.005)
                    continue
                for cam_id in due:
                    try:
                        self._process_camera(cam_id, now)
                    except Exception:
                        logger.exception("cam=%s processing error", cam_id)
                    self.scheduler.mark_done(cam_id, time.monotonic())
        finally:
            self.stop()
