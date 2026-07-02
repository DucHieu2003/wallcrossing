"""Chan doan kha nang decode video tren box RK3588.

Chay tren box:
    python3 tools/check_decode.py                          # chi kiem tra moi truong
    python3 tools/check_decode.py "rtsp://user:pass@ip:554/ch01/0"   # + do fps thuc te

Script se cho biet nen dat DECODE_BACKEND / FFMPEG_VIDEO_CODEC nao trong config.py.
"""
from __future__ import annotations

import subprocess
import sys
import time

import cv2


def check_opencv_gstreamer() -> bool:
    for line in cv2.getBuildInformation().splitlines():
        if "GStreamer" in line:
            ok = "YES" in line
            print(f"[{'OK' if ok else 'NO'}] OpenCV built with GStreamer: {line.strip()}")
            return ok
    print("[NO] OpenCV build info has no GStreamer line")
    return False


def check_gst_mpp() -> bool:
    try:
        r = subprocess.run(
            ["gst-inspect-1.0", "mppvideodec"],
            capture_output=True, timeout=10,
        )
        ok = r.returncode == 0
        print(f"[{'OK' if ok else 'NO'}] GStreamer element mppvideodec (HW decode)")
        return ok
    except FileNotFoundError:
        print("[NO] gst-inspect-1.0 not found (GStreamer not installed)")
        return False
    except subprocess.TimeoutExpired:
        print("[NO] gst-inspect-1.0 timed out")
        return False


def check_ffmpeg_rkmpp() -> list[str]:
    try:
        r = subprocess.run(["ffmpeg", "-decoders"], capture_output=True, text=True, timeout=10)
    except FileNotFoundError:
        print("[NO] ffmpeg binary not found")
        return []
    found = [
        line.split()[1]
        for line in r.stdout.splitlines()
        if "rkmpp" in line and len(line.split()) > 1
    ]
    if found:
        print(f"[OK] ffmpeg co decoder rkmpp: {', '.join(found)}")
        print("     LUU Y: neu OpenCV cai bang pip (opencv-python), no dung FFmpeg")
        print("     rieng ben trong — chua chac dung duoc decoder nay. Chay test RTSP")
        print("     ben duoi de xac nhan.")
    else:
        print("[NO] ffmpeg khong co decoder rkmpp")
    return found


def bench_capture(name: str, open_fn, seconds: float = 6.0) -> None:
    cap = open_fn()
    if not cap.isOpened():
        print(f"  {name}: KHONG MO DUOC")
        return
    # bo qua frame dau (khoi dong decoder)
    cap.read()
    n = 0
    size = None
    t0 = time.monotonic()
    while time.monotonic() - t0 < seconds:
        ok, frame = cap.read()
        if ok and frame is not None:
            n += 1
            size = (frame.shape[1], frame.shape[0])
    cap.release()
    fps = n / seconds
    print(f"  {name}: {fps:.1f} fps, frame size = {size}")


def bench_rtsp(url: str, gst_ok: bool, rkmpp_decoders: list[str]) -> None:
    import os

    print(f"\n=== Test doc RTSP (moi test ~6s): {url.split('@')[-1]} ===")

    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
    bench_capture("FFmpeg (sw decode)", lambda: cv2.VideoCapture(url, cv2.CAP_FFMPEG))

    for dec in rkmpp_decoders:
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = f"rtsp_transport;tcp|video_codec;{dec}"
        bench_capture(
            f"FFmpeg ({dec})", lambda: cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        )
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

    if gst_ok:
        for codec, depay in [("h265", "rtph265depay ! h265parse"), ("h264", "rtph264depay ! h264parse")]:
            pipe = (
                f"rtspsrc location={url} latency=200 protocols=tcp ! "
                f"{depay} ! mppvideodec ! videoconvert ! video/x-raw,format=BGR ! "
                "appsink drop=true max-buffers=1 sync=false"
            )
            bench_capture(
                f"GStreamer mppvideodec ({codec})",
                lambda p=pipe: cv2.VideoCapture(p, cv2.CAP_GSTREAMER),
            )


def main() -> None:
    print("=== Moi truong ===")
    print(f"OpenCV {cv2.__version__}")
    gst_ok = check_opencv_gstreamer()
    mpp_ok = check_gst_mpp()
    rkmpp = check_ffmpeg_rkmpp()

    print("\n=== Ket luan so bo ===")
    if gst_ok and mpp_ok:
        print("-> Dung DECODE_BACKEND = \"gstreamer\" (HW decode qua mppvideodec).")
    elif rkmpp:
        print("-> Thu FFMPEG_VIDEO_CODEC = \"hevc_rkmpp\" voi DECODE_BACKEND = \"opencv\".")
        print("   Neu test RTSP ben duoi khong mo duoc -> OpenCV pip khong thay duoc")
        print("   ffmpeg he thong; can cai opencv build voi GStreamer, hoac dung substream.")
    else:
        print("-> Khong co duong decode phan cung nao san.")
        print("   Cach nhanh nhat: doi RTSP sang SUBSTREAM (/ch01/1) va/hoac vao web")
        print("   camera doi encoding H.265 -> H.264, giam fps xuong 10-15.")

    if len(sys.argv) > 1:
        bench_rtsp(sys.argv[1], gst_ok and mpp_ok, rkmpp)
    else:
        print("\n(Truyen them RTSP URL de do fps thuc te tung phuong an.)")


if __name__ == "__main__":
    main()
