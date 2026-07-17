# wallcrossing

Pipeline phát hiện người chạm/vượt vùng tường từ luồng RTSP camera trên RK3588 AIBox.

## Pipeline

```text
RTSP camera
  -> RtspReader giữ frame mới nhất
  -> FrameScheduler chọn camera theo detect_fps
  -> RKNN chạy YOLO26s trên NPU
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

## Thu thập dataset từ camera thật

Pipeline lưu ảnh full-frame và nhãn YOLO tương ứng để fine-tune:

```text
outputs/dataset_capture/
  images/detections/YYYY-MM-DD/<camera>/...jpg
  labels/detections/YYYY-MM-DD/<camera>/...txt
  images/background/YYYY-MM-DD/<camera>/...jpg
  labels/background/YYYY-MM-DD/<camera>/...txt
  images/hard_negatives/YYYY-MM-DD/<camera>/...jpg
  labels/hard_negatives/YYYY-MM-DD/<camera>/...txt
  metadata.jsonl
```

Nhãn detection có dạng `class_id center_x center_y width height`, chuẩn hóa theo
full frame. Background có nhãn rỗng. Hard-negative là vật tĩnh mà model nhận nhầm
thành người (ví dụ bạt), được motion filter giữ thưa với nhãn rỗng. Dataset có
quota rolling; vượt giới hạn thì xóa đồng thời cặp ảnh/nhãn cũ nhất.

Các tham số trong `config.py`:

```python
DATASET_CAPTURE_ENABLED = True
DATASET_DETECTION_INTERVAL_SECONDS = 5.0
DATASET_BACKGROUND_INTERVAL_SECONDS = 28800.0
DATASET_HARD_NEGATIVE_INTERVAL_SECONDS = 21600.0
DATASET_MAX_DISK_GB = 20.0
MOTION_FILTER_ENABLED = True
MOTION_MIN_RATIO = 0.05
```

Ảnh dataset là raw frame, không vẽ bbox/polygon. `metadata.jsonl` giữ thêm
confidence, overlap và đường dẫn; file này tự rotate ở 50 MB.

## Giới hạn output và RAM

```python
EVIDENCE_MAX_DISK_GB = 2.0
DEBUG_PREVIEW_MAX_DISK_GB = 1.0
ALERT_LOG_MAX_MB = 10
ALERT_LOG_BACKUP_COUNT = 2
RSS_GRACEFUL_RESTART_MB = 4800
RSS_CHECK_INTERVAL_SECONDS = 5.0
```

Evidence/debug preview tự xóa JPG cũ nhất khi vượt quota. Log service dùng
rotation 10 MB x 6 file; `alerts.jsonl` dùng 10 MB x 3 file. Khi RSS đạt 4.8 GB,
ứng dụng dừng mềm để `Restart=always` của systemd khởi động lại; cgroup vẫn có
ngưỡng mềm 5.5 GB và giới hạn cứng 6.5 GB.

## Chạy bằng systemd trên box RK3588

Unit `deploy/wallcrossing.service` tự chạy khi boot, tự restart khi process crash,
bị kill hoặc bị cgroup OOM kill. Với box 8 GB, unit dùng ngưỡng mềm 5.5 GB và
giới hạn cứng 6.5 GB để chừa RAM cho hệ điều hành, VPU/NPU và SSH.

Cài và khởi động:

```bash
sudo cp deploy/wallcrossing.service /etc/systemd/system/wallcrossing.service
sudo systemd-analyze verify /etc/systemd/system/wallcrossing.service
sudo systemctl daemon-reload
sudo systemctl enable --now wallcrossing.service
```

Xem trạng thái và số lần restart:

```bash
systemctl status wallcrossing.service --no-pager
systemctl show wallcrossing.service \
  -p ActiveState -p SubState -p Result -p NRestarts
```

Xem RAM hiện tại và giới hạn:

```bash
systemctl show wallcrossing.service \
  -p MemoryCurrent -p MemoryHigh -p MemoryMax
cat /sys/fs/cgroup/system.slice/wallcrossing.service/memory.current
cat /sys/fs/cgroup/system.slice/wallcrossing.service/memory.events
```

Trong `memory.events`, theo dõi các bộ đếm `high`, `max`, `oom` và `oom_kill`.
Service cũng ghi `rss_mb`, `hwm_mb`, số reader connected/stale và reconnect mỗi
`HEALTH_LOG_INTERVAL_SECONDS`.

Xem log đầy đủ (file tự rotate 10 MB x 6):

```bash
tail -f logs/wallcrossing_service.log
```

Journal chỉ giữ stderr/lỗi khởi động để tránh RTSP reconnect spam chiếm quota hệ thống:

```bash
journalctl -u wallcrossing.service -f
```

Restart hoặc stop thủ công:

```bash
sudo systemctl restart wallcrossing.service
sudo systemctl stop wallcrossing.service
```

`systemctl stop` giữ service ở trạng thái dừng. `Restart=always` chỉ khởi động lại
sau lỗi hoặc signal không phải thao tác stop có chủ đích của systemd.

Kiểm tra thời gian shutdown:

```bash
time sudo systemctl stop wallcrossing.service
sudo systemctl start wallcrossing.service
```

Ứng dụng có deadline shutdown 15 giây; systemd cưỡng chế dừng sau 20 giây nếu
thread native của OpenCV/GStreamer không thoát.
