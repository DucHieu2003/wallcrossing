from pathlib import Path

from wallcrossing.alerts.alerting import AlertManager
from wallcrossing.core.models import AlertEvent


def _mgr(tmp_path: Path, hits=2, cooldown=30.0) -> AlertManager:
    return AlertManager(consecutive_hits=hits, cooldown_seconds=cooldown, alert_log_path=tmp_path / "alerts.jsonl")


def test_requires_consecutive_hits(tmp_path):
    m = _mgr(tmp_path, hits=3)
    assert m.update("cam", True, now_mono=0.0) is False
    assert m.update("cam", True, now_mono=1.0) is False
    assert m.update("cam", True, now_mono=2.0) is True


def test_streak_resets_on_miss(tmp_path):
    m = _mgr(tmp_path, hits=2)
    assert m.update("cam", True, now_mono=0.0) is False
    assert m.update("cam", False, now_mono=1.0) is False  # reset
    assert m.update("cam", True, now_mono=2.0) is False  # streak back to 1
    assert m.update("cam", True, now_mono=3.0) is True


def test_cooldown_suppresses_duplicate(tmp_path):
    m = _mgr(tmp_path, hits=1, cooldown=30.0)
    assert m.update("cam", True, now_mono=100.0) is True
    assert m.update("cam", True, now_mono=110.0) is False  # within cooldown
    assert m.update("cam", True, now_mono=131.0) is True  # cooldown elapsed


def test_per_camera_independent(tmp_path):
    m = _mgr(tmp_path, hits=1, cooldown=30.0)
    assert m.update("a", True, now_mono=0.0) is True
    assert m.update("b", True, now_mono=0.0) is True


def _event():
    return AlertEvent(
        alert_id="cam-1",
        camera_id="cam",
        camera_name="Gate",
        timestamp="2026-06-30T10:00:00.000Z",
        bbox_xyxy=(1, 2, 3, 4),
        confidence=0.9,
        wall_polygon=[[0, 0], [1, 0], [1, 1]],
        overlap_ratio=0.5,
        evidence_path="x.jpg",
    )


def test_write_appends_jsonl(tmp_path):
    m = _mgr(tmp_path, hits=1)
    m.write(_event())
    m.write(_event())
    lines = (tmp_path / "alerts.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2


def test_alert_log_rotation_keeps_configured_backups(tmp_path):
    path = tmp_path / "alerts.jsonl"
    m = AlertManager(1, 0.0, path, log_max_mb=1, log_backup_count=2)
    # Use a tiny threshold after initialization for a fast deterministic test.
    m.log_max_bytes = 1

    m.write(_event())
    m.write(_event())
    m.write(_event())
    m.write(_event())

    assert path.exists()
    assert (tmp_path / "alerts.jsonl.1").exists()
    assert (tmp_path / "alerts.jsonl.2").exists()
    assert not (tmp_path / "alerts.jsonl.3").exists()
