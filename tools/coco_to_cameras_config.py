"""Generate CAMERA_CONFIGS Python snippet from Roboflow COCO wall annotations.

Offline helper only. Runtime does not read the dataset; it reads CAMERA_CONFIGS in
root config.py / config.local.py.

Example:
  python tools/coco_to_cameras_config.py --out config.local.generated.py
"""

from __future__ import annotations

import argparse
import json
import pprint
import re
from pathlib import Path


def _camera_id_from_name(name: str) -> str:
    stem = Path(name).stem
    ip = re.search(r"(\d+)[-_](\d+)[-_](\d+)[-_](\d+)", stem)
    if ip:
        return "cam_" + "_".join(ip.groups())
    return "cam_" + re.sub(r"\W+", "_", stem).strip("_").lower()


def _rtsp_url_from_camera_id(camera_id: str) -> str:
    parts = camera_id.replace("cam_", "").split("_")
    if len(parts) == 4 and all(p.isdigit() for p in parts):
        return f"rtsp://user:pass@{'.'.join(parts)}:554/stream"
    return "rtsp://user:pass@host:554/stream"


def _first_segmentation(annotation: dict) -> list[list[float]] | None:
    seg = annotation.get("segmentation")
    if not seg:
        return None
    flat = seg[0]
    if len(flat) < 6:
        return None
    return [[round(float(flat[i]), 2), round(float(flat[i + 1]), 2)] for i in range(0, len(flat), 2)]


def build_camera_configs(annotations_path: Path) -> list[dict]:
    coco = json.loads(annotations_path.read_text(encoding="utf-8"))
    images = {img["id"]: img for img in coco["images"]}
    anns_by_image: dict[int, list[dict]] = {}
    for ann in coco["annotations"]:
        anns_by_image.setdefault(ann["image_id"], []).append(ann)

    cameras = []
    for image_id, anns in sorted(anns_by_image.items()):
        img = images[image_id]
        polygon = _first_segmentation(anns[0])
        if not polygon:
            continue
        source = img.get("extra", {}).get("name", img["file_name"])
        camera_id = _camera_id_from_name(source)
        cameras.append(
            {
                "id": camera_id,
                "name": source,
                "rtsp_url": _rtsp_url_from_camera_id(camera_id),
                "enabled": True,
                "wall_polygon": polygon,
            }
        )
    return cameras


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate CAMERA_CONFIGS from Roboflow COCO wall annotations")
    parser.add_argument("--annotations", default="wall.v1i.coco/train/_annotations.coco.json")
    parser.add_argument("--out", default="config.local.generated.py")
    args = parser.parse_args()

    cameras = build_camera_configs(Path(args.annotations))
    out = Path(args.out)
    content = (
        "# Generated from Roboflow COCO annotations.\n"
        "# Review RTSP credentials and camera assignment before deploy.\n\n"
        f"CAMERA_CONFIGS = {pprint.pformat(cameras, width=120, sort_dicts=False)}\n"
    )
    out.write_text(content, encoding="utf-8")
    print(f"wrote {out} ({len(cameras)} cameras)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
