from __future__ import annotations

import logging
from collections import deque
from pathlib import Path

logger = logging.getLogger("wallcrossing.storage_quota")


class DirectoryQuota:
    """Rolling quota that removes the oldest matching files first."""

    def __init__(self, root_dir: str | Path, max_disk_gb: float, pattern: str = "*.jpg"):
        self.root_dir = Path(root_dir)
        self.max_bytes = int(max_disk_gb * 1024**3) if max_disk_gb > 0 else 0
        self.pattern = pattern
        self._tracked: deque[tuple[Path, int]] = deque()
        self._tracked_bytes = 0
        if self.max_bytes:
            self._scan_existing()
            self._enforce()

    @property
    def tracked_bytes(self) -> int:
        return self._tracked_bytes

    def _scan_existing(self) -> None:
        files: list[tuple[float, Path, int]] = []
        for path in self.root_dir.rglob(self.pattern):
            try:
                stat = path.stat()
            except OSError:
                continue
            files.append((stat.st_mtime, path, stat.st_size))
        files.sort(key=lambda item: item[0])
        for _, path, size in files:
            self._tracked.append((path, size))
            self._tracked_bytes += size

    def track(self, path: str | Path) -> None:
        if not self.max_bytes:
            return
        tracked_path = Path(path)
        try:
            size = tracked_path.stat().st_size
        except OSError:
            return
        self._tracked.append((tracked_path, size))
        self._tracked_bytes += size
        self._enforce()

    def _enforce(self) -> None:
        removed = 0
        while self._tracked_bytes > self.max_bytes and self._tracked:
            path, size = self._tracked.popleft()
            self._tracked_bytes -= size
            try:
                path.unlink()
            except FileNotFoundError:
                pass
            except OSError:
                logger.warning("failed to remove old quota file: %s", path)
            removed += 1
        if removed:
            logger.info(
                "quota dir=%s removed=%d usage=%.2fGB limit=%.2fGB",
                self.root_dir,
                removed,
                self._tracked_bytes / 1024**3,
                self.max_bytes / 1024**3,
            )
