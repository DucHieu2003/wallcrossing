import os

from wallcrossing.runtime.storage_quota import DirectoryQuota


def test_directory_quota_removes_oldest_files(tmp_path):
    old = tmp_path / "old.jpg"
    new = tmp_path / "new.jpg"
    old.write_bytes(b"a" * 100)
    new.write_bytes(b"b" * 100)
    os.utime(old, (1, 1))
    os.utime(new, (2, 2))

    quota = DirectoryQuota(tmp_path, max_disk_gb=1.0)
    quota.max_bytes = 150
    quota._enforce()

    assert not old.exists()
    assert new.exists()
    assert quota.tracked_bytes == 100


def test_directory_quota_tracks_new_file_and_enforces_limit(tmp_path):
    quota = DirectoryQuota(tmp_path, max_disk_gb=1.0)
    quota.max_bytes = 100
    first = tmp_path / "first.jpg"
    second = tmp_path / "second.jpg"
    first.write_bytes(b"a" * 60)
    second.write_bytes(b"b" * 60)

    quota.track(first)
    quota.track(second)

    assert not first.exists()
    assert second.exists()
    assert quota.tracked_bytes == 60
