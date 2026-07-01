# wallcrossing

Pipeline phát hiện người chạm/vượt vùng tường từ luồng RTSP camera trên RK3588 AIBox.

## Pipeline

```text
RTSP camera
  -> RtspReader giữ frame mới nhất
  -> FrameScheduler chọn camera theo detect_fps
  -> RKNNLite chạy YOLO26s trên NPU
  -> postprocess decode YOLO raw-head output [1, 84, 8400]
  -> wall_contact tính overlap phần chân người với wall_polygon
  -> AlertManager lọc consecutive_hits + cooldown
  -> lưu evidence JPG + append alerts.jsonl
```

## Cấu trúc

```text
main.py                         entrypoint service
config.py                       cấu hình camera, model, rule, output
wallcrossing/core/              schema config + model dữ liệu
wallcrossing/streams/           đọc RTSP bằng OpenCV/GStreamer
wallcrossing/detection/         detector backend + postprocess YOLO
wallcrossing/services/          logic chạm vùng tường
wallcrossing/alerts/            chống spam alert + lưu evidence
wallcrossing/runtime/           pipeline chính + scheduler
wallcrossing/utils/             logging setup
tools/convert_rknn.py           export ONNX raw-head và convert RKNN
tests/                          unit tests
```

## Cấu hình chính

Sửa trực tiếp trong `config.py`:

```python
MODEL_BACKEND = "rknn"
DECODE_BACKEND = "gstreamer"
YOLO26_RKNN_PATH = "weights/yolo26s_rk3588_fp16.rknn"
DEFAULT_DETECT_FPS = 5
```

Mỗi camera trong `CAMERA_CONFIGS` cần:

```python
{
    "id": "cam_...",
    "name": "...",
    "rtsp_url": "rtsp://...",
    "enabled": True,
    "wall_polygon": [[x, y], ...],
}
```

## Rule cảnh báo

```python
MIN_OVERLAP_RATIO = 0.02
CONSECUTIVE_HITS = 2
COOLDOWN_SECONDS = 30
CONTACT_MODE = "bottom_band"
BOTTOM_BAND_RATIO = 0.25
```

`bottom_band` chỉ dùng phần dưới bbox người để kiểm tra chân có chạm vùng tường hay không.

## Chạy service

```bash
python main.py --log-level INFO
```

Output:

```text
logs/wallcrossing_service.log       log service
logs/alerts.jsonl                   alert event dạng JSONL
outputs/evidence/YYYY-MM-DD/...jpg  ảnh bằng chứng có vẽ polygon + bbox
```
