from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Optional

from wallcrossing.core.models import AlertEvent

logger = logging.getLogger("wallcrossing.alerting")


class AlertManager:
    """Decides when a wall-contact candidate becomes a real alert.

    A detection must contact the wall for `consecutive_hits` frames in a row
    before it fires, and the same camera is muted for `cooldown_seconds` after
    a fire to avoid spamming.
    """

    def __init__(
        self,
        consecutive_hits: int,
        cooldown_seconds: float,
        alert_log_path: str | Path,
    ):
        self.consecutive_hits = consecutive_hits
        self.cooldown_seconds = cooldown_seconds
        self.alert_log_path = Path(alert_log_path)
        self.alert_log_path.parent.mkdir(parents=True, exist_ok=True)

        self._hit_streak: dict[str, int] = defaultdict(int)
        self._last_alert_mono: dict[str, float] = {}

    def update(
        self,
        camera_id: str,
        contacted: bool,
        now_mono: float,
    ) -> bool:
        """Feed one frame's contact result for a camera. Returns True if it should fire now."""
        if not contacted:
            self._hit_streak[camera_id] = 0
            return False

        self._hit_streak[camera_id] += 1
        if self._hit_streak[camera_id] < self.consecutive_hits:
            return False

        last = self._last_alert_mono.get(camera_id)
        if last is not None and (now_mono - last) < self.cooldown_seconds:
            return False

        self._last_alert_mono[camera_id] = now_mono
        self._hit_streak[camera_id] = 0
        return True

    def write(self, event: AlertEvent) -> None:
        with self.alert_log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event.to_log_dict(), ensure_ascii=False) + "\n")
        logger.info(
            "ALERT %s cam=%s conf=%.2f overlap=%.3f -> %s",
            event.alert_id,
            event.camera_id,
            event.confidence,
            event.overlap_ratio,
            event.evidence_path,
        )
