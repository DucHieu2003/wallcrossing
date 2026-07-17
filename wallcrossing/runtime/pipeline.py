from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np

from wallcrossing.alerts.alerting import AlertManager
from wallcrossing.alerts.evidence import (
    debug_preview_path,
    draw_and_save,
    draw_detections_preview,
    evidence_path,
)
from wallcrossing.core.config_loader import AppConfig
from wallcrossing.core.models import AlertEvent, Detection
from wallcrossing.detection.detector import build_detector
from wallcrossing.services.wall_contact import touches_wall
from wallcrossing.runtime.dataset_capture import DatasetCapture
from wallcrossing.runtime.motion_filter import MotionFilter
from wallcrossing.runtime.roi import roi_from_polygon, resolve_roi_axis, translate_detection
from wallcrossing.runtime.scheduler import FrameScheduler
from wallcrossing.runtime.storage_quota import DirectoryQuota
from wallcrossing.runtime.systemd_notify import notify_systemd
from wallcrossing.streams.rtsp_reader import RtspReader

logger = logging.getLogger("wallcrossing.pipeline")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def _process_memory_mb() -> tuple[float, float]:
    values: dict[str, float] = {}
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            key, separator, value = line.partition(":")
            if separator and key in {"VmRSS", "VmHWM"}:
                values[key] = float(value.strip().split()[0]) / 1024.0
    except (OSError, ValueError, IndexError):
        return 0.0, 0.0
    return values.get("VmRSS", 0.0), values.get("VmHWM", 0.0)


class Pipeline:
    def __init__(
        self,
        cfg: AppConfig,
        notify_systemd: Callable[[str], None] = notify_systemd,
    ):
        self.cfg = cfg
        self._notify_systemd = notify_systemd
        self.detector = build_detector(cfg.model)
        self.alerts = AlertManager(
            consecutive_hits=cfg.rules.consecutive_hits,
            cooldown_seconds=cfg.rules.cooldown_seconds,
            alert_log_path=cfg.pipeline.alert_log_path,
            log_max_mb=cfg.pipeline.alert_log_max_mb,
            log_backup_count=cfg.pipeline.alert_log_backup_count,
        )
        self.evidence_quota = DirectoryQuota(
            cfg.pipeline.evidence_dir, cfg.pipeline.evidence_max_disk_gb
        )
        self.debug_preview_quota = DirectoryQuota(
            cfg.pipeline.debug_preview_dir, cfg.pipeline.debug_preview_max_disk_gb
        )
        self.dataset_capture = (
            DatasetCapture(
                root_dir=cfg.pipeline.dataset_capture_dir,
                detection_interval_seconds=cfg.pipeline.dataset_detection_interval_seconds,
                background_interval_seconds=cfg.pipeline.dataset_background_interval_seconds,
                jpeg_quality=cfg.pipeline.dataset_jpeg_quality,
                max_disk_gb=cfg.pipeline.dataset_max_disk_gb,
                hard_negative_interval_seconds=(
                    cfg.pipeline.dataset_hard_negative_interval_seconds
                ),
            )
            if cfg.pipeline.dataset_capture_enabled
            else None
        )
        self.motion_filter = (
            MotionFilter(min_motion_ratio=cfg.pipeline.motion_min_ratio)
            if cfg.pipeline.motion_filter_enabled
            else None
        )
        self.cameras = {c.id: c for c in cfg.enabled_cameras()}
        self.readers: dict[str, RtspReader] = {
            c.id: RtspReader(
                c.id,
                c.rtsp_url,
                decode_backend=cfg.pipeline.decode_backend,
                target_fps=cfg.detect_fps_for(c),
                codec=cfg.pipeline.codec,
                transport=cfg.pipeline.rtsp_transport,
                ffmpeg_video_codec=cfg.pipeline.ffmpeg_video_codec,
                initial_delay=i * 0.5,
            )
            for i, c in enumerate(self.cameras.values())
        }
        self.scheduler = FrameScheduler(
            {c.id: cfg.detect_fps_for(c) for c in self.cameras.values()}
        )
        self._last_index: dict[str, int] = {}
        self._last_debug_preview_mono: dict[str, float] = {}
        # cam_id -> ((frame_h, frame_w), polygon da scale theo kich thuoc frame)
        self._scaled_polygons: dict[str, tuple[tuple[int, int], list[list[float]]]] = {}
        self._logged_roi_shapes: set[tuple[str, tuple[int, int]]] = set()
        self._stop = False
        self._stopped = False
        self._started_mono = time.monotonic()
        self._next_health_log_mono = self._started_mono + cfg.pipeline.health_log_interval_seconds
        self._next_rss_check_mono = self._started_mono
        self._next_watchdog_mono = self._started_mono
        self._stale_logged: set[str] = set()
        self._process_errors: dict[str, int] = {cam_id: 0 for cam_id in self.cameras}

    def _wall_polygon_for(self, cam, frame_shape: tuple[int, int]) -> list[list[float]]:
        """Wall polygon in the coordinate space of the actual frame.

        Polygons are drawn on reference images (polygon_ref_size); when the RTSP
        frame has a different resolution (e.g. substream) they are scaled once
        and cached per frame size.
        """
        cached = self._scaled_polygons.get(cam.id)
        if cached is not None and cached[0] == frame_shape:
            return cached[1]

        h, w = frame_shape
        if cam.polygon_ref_size:
            ref_w, ref_h = cam.polygon_ref_size
            sx, sy = w / ref_w, h / ref_h
            polygon = [[x * sx, y * sy] for x, y in cam.wall_polygon]
            if abs(sx / sy - 1.0) > 0.02:
                logger.warning(
                    "cam=%s frame %dx%d khac ty le khung hinh voi polygon_ref_size %sx%s "
                    "- polygon co the bi meo, nen ve lai tren dung do phan giai",
                    cam.id, w, h, ref_w, ref_h,
                )
        else:
            polygon = cam.wall_polygon

        self._scaled_polygons[cam.id] = (frame_shape, polygon)
        return polygon

    def start(self) -> None:
        for r in self.readers.values():
            r.start()
        logger.info("da khoi dong %d camera reader", len(self.readers))
        self._notify_systemd("READY=1")

    def _notify_watchdog(self) -> None:
        self._notify_systemd("WATCHDOG=1")

    def request_stop(self) -> None:
        self._stop = True

    def stop(self) -> None:
        if self._stopped:
            return
        self._stopped = True
        self.request_stop()
        self._notify_systemd("STOPPING=1")
        logger.info("dang dung %d camera reader", len(self.readers))
        for reader in self.readers.values():
            reader.request_stop()

        deadline = time.monotonic() + self.cfg.pipeline.shutdown_timeout_seconds
        stuck: list[str] = []
        for cam_id, reader in self.readers.items():
            remaining = max(0.0, deadline - time.monotonic())
            if not reader.join(timeout=remaining):
                stuck.append(cam_id)
        if stuck:
            logger.warning("reader thread chua dung sau thoi han: %s", ", ".join(stuck))

        try:
            self.detector.close()
        except Exception:
            logger.exception("don dep detector that bai")
        logger.info("pipeline da dung hoan toan")

    def _check_memory(self, now_mono: float) -> None:
        if now_mono < self._next_rss_check_mono:
            return
        self._next_rss_check_mono = now_mono + self.cfg.pipeline.rss_check_interval_seconds
        restart_at = self.cfg.pipeline.rss_graceful_restart_mb
        if restart_at <= 0:
            return
        rss_mb, _ = _process_memory_mb()
        if rss_mb >= restart_at:
            logger.critical(
                "RSS %.1fMB dat nguong restart mem %.1fMB; dang dung de systemd khoi dong lai",
                rss_mb,
                restart_at,
            )
            self.request_stop()

    def _log_health(self, now_mono: float) -> None:
        if now_mono < self._next_health_log_mono:
            return
        self._next_health_log_mono = now_mono + self.cfg.pipeline.health_log_interval_seconds
        stale_after = self.cfg.pipeline.stale_reader_warn_seconds
        stale: list[str] = []
        connected = 0
        reconnects = 0
        for cam_id, reader in self.readers.items():
            connected += int(reader.connected)
            reconnects += reader.reconnect_count
            frame_age = now_mono - reader.last_frame_mono if reader.last_frame_mono else now_mono - self._started_mono
            if frame_age >= stale_after:
                stale.append(cam_id)
                if cam_id not in self._stale_logged:
                    logger.warning("cam=%s khong co frame moi trong %.1fs", cam_id, frame_age)
                    self._stale_logged.add(cam_id)
            else:
                self._stale_logged.discard(cam_id)

        rss_mb, hwm_mb = _process_memory_mb()
        logger.info(
            "suc_khoe rss_mb=%.1f hwm_mb=%.1f readers=%d ket_noi=%d tre=%d reconnect=%d",
            rss_mb,
            hwm_mb,
            len(self.readers),
            connected,
            len(stale),
            reconnects,
        )

    def _save_debug_preview(self, cam, cam_id, frame, wall_polygon, detections, now_mono) -> None:
        interval = self.cfg.pipeline.debug_preview_interval_seconds
        last = self._last_debug_preview_mono.get(cam_id)
        if last is not None and now_mono - last < interval:
            return
        self._last_debug_preview_mono[cam_id] = now_mono

        ts = _utc_now_iso()
        out_path = debug_preview_path(self.cfg.pipeline.debug_preview_dir, cam_id, ts)
        confs = ",".join(f"{det.confidence:.2f}" for det in detections)
        label = f"{cam.name or cam_id} n={len(detections)} conf={confs}"
        draw_detections_preview(frame, wall_polygon, detections, label, out_path)
        self.debug_preview_quota.track(out_path)

    def _detect(self, cam, cam_id: str, frame, wall_polygon) -> list[Detection]:
        if self.cfg.detect_roi_enabled_for(cam):
            x1, y1, x2, y2 = roi_from_polygon(
                wall_polygon,
                frame.shape[:2],
                self.cfg.detect_roi_min_extent_ratio_for(cam),
                self.cfg.detect_roi_side_margin_ratio_for(cam),
                self.cfg.detect_roi_axis_for(cam),
            )
            crop = frame[y1:y2, x1:x2]
            detections = [
                translate_detection(det, x1, y1)
                for det in self.detector.detect(crop)
            ]
            roi_key = (cam_id, frame.shape[:2])
            if roi_key not in self._logged_roi_shapes:
                self._logged_roi_shapes.add(roi_key)
                logger.info(
                    "cam=%s detect_roi=(%d,%d,%d,%d) frame=%dx%d axis=%s",
                    cam_id,
                    x1,
                    y1,
                    x2,
                    y2,
                    frame.shape[1],
                    frame.shape[0],
                    resolve_roi_axis(
                        wall_polygon,
                        self.cfg.detect_roi_axis_for(cam),
                        frame.shape[:2],
                    ),
                )
        else:
            detections = self.detector.detect(frame)

        if detections:
            logger.info(
                "cam=%s so_nguoi=%d do_tin_cay=%s",
                cam_id,
                len(detections),
                ",".join(f"{det.confidence:.2f}" for det in detections),
            )
        return detections

    def _evaluate_contacts(
        self,
        detections: list[Detection],
        wall_polygon: list[list[float]],
    ) -> tuple[Detection | None, float, list[float]]:
        wall = np.asarray(wall_polygon, dtype=float)
        best: Detection | None = None
        best_ratio = 0.0
        overlap_ratios: list[float] = []
        for det in detections:
            hit, ratio = touches_wall(
                det.bbox_xyxy,
                wall,
                self.cfg.rules.min_overlap_ratio,
                self.cfg.rules.contact_mode,
                self.cfg.rules.bottom_band_ratio,
            )
            overlap_ratios.append(ratio)
            if hit and ratio > best_ratio:
                best = det
                best_ratio = ratio
        return best, best_ratio, overlap_ratios

    def _capture_dataset(
        self,
        cam_id: str,
        frame,
        detections: list[Detection],
        overlap_ratios: list[float],
        now_mono: float,
    ) -> None:
        if self.dataset_capture is None:
            return

        capture_detections = detections
        capture_overlaps = overlap_ratios
        motion_ready = True
        if self.motion_filter is not None:
            capture_detections, capture_overlaps = self.motion_filter.filter(
                cam_id, frame, detections, overlap_ratios
            )
            motion_ready = self.motion_filter.is_ready(cam_id)
            if motion_ready and detections and not capture_detections:
                logger.info(
                    "cam=%s bo %d detection tinh (khong co chuyen dong)",
                    cam_id,
                    len(detections),
                )
        if detections and not motion_ready:
            return

        self.dataset_capture.capture(
            camera_id=cam_id,
            timestamp=_utc_now_iso(),
            now_mono=now_mono,
            frame=frame,
            detections=capture_detections,
            overlap_ratios=capture_overlaps,
            hard_negative=bool(detections and not capture_detections),
        )

    def _fire_alert(
        self,
        cam,
        cam_id: str,
        frame,
        wall_polygon: list[list[float]],
        best: Detection | None,
        best_ratio: float,
        now_mono: float,
    ) -> None:
        if not self.cfg.pipeline.alerts_enabled:
            return
        should_fire = self.alerts.update(cam_id, best is not None, now_mono)
        if not should_fire or best is None:
            return

        ts = _utc_now_iso()
        alert_id = f"{cam_id}-{ts.replace(':', '').replace('-', '').replace('.', '')}"
        out_path = evidence_path(self.cfg.pipeline.evidence_dir, cam_id, ts, alert_id)
        label = f"{cam.name or cam_id} {best.confidence:.2f} {ts}"
        draw_and_save(frame, wall_polygon, best, label, out_path)
        self.evidence_quota.track(out_path)

        event = AlertEvent(
            alert_id=alert_id,
            camera_id=cam_id,
            camera_name=cam.name,
            timestamp=ts,
            bbox_xyxy=best.bbox_xyxy,
            confidence=best.confidence,
            wall_polygon=wall_polygon,
            overlap_ratio=best_ratio,
            evidence_path=str(out_path),
        )
        self.alerts.write(event)

    def _process_camera(self, cam_id: str, now_mono: float) -> bool:
        cam = self.cameras[cam_id]
        frame, index = self.readers[cam_id].read_latest()
        if frame is None or self._last_index.get(cam_id) == index:
            return False
        self._last_index[cam_id] = index

        wall_polygon = self._wall_polygon_for(cam, frame.shape[:2])
        detections = self._detect(cam, cam_id, frame, wall_polygon)
        best, best_ratio, overlap_ratios = self._evaluate_contacts(
            detections, wall_polygon
        )

        if detections and self.cfg.pipeline.debug_preview_enabled:
            self._save_debug_preview(
                cam, cam_id, frame, wall_polygon, detections, now_mono
            )

        self._capture_dataset(
            cam_id, frame, detections, overlap_ratios, now_mono
        )
        self._fire_alert(
            cam, cam_id, frame, wall_polygon, best, best_ratio, now_mono
        )
        return True

    def run(self) -> None:
        self.start()
        try:
            while not self._stop:
                now = time.monotonic()
                if now >= self._next_watchdog_mono:
                    self._notify_watchdog()
                    self._next_watchdog_mono = now + 20.0
                self._check_memory(now)
                self._log_health(now)
                if self._stop:
                    break
                due = self.scheduler.due_cameras(now)
                if not due:
                    time.sleep(0.005)
                    continue
                for cam_id in due:
                    try:
                        processed = self._process_camera(cam_id, now)
                        if processed:
                            self._process_errors[cam_id] = 0
                    except Exception:
                        errors = self._process_errors.get(cam_id, 0) + 1
                        self._process_errors[cam_id] = errors
                        logger.exception(
                            "cam=%s loi xu ly (%d/%d)",
                            cam_id,
                            errors,
                            self.cfg.pipeline.max_consecutive_process_errors,
                        )
                        if errors >= self.cfg.pipeline.max_consecutive_process_errors:
                            raise RuntimeError(
                                f"cam={cam_id} vuot qua gioi han loi xu ly lien tiep"
                            )
                    self.scheduler.mark_done(cam_id, time.monotonic())
        finally:
            self.stop()
