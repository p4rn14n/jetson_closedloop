import os
import time
import argparse
import configparser
import csv
from datetime import datetime
import cv2

from picamera2 import Picamera2
from picamera2.encoders import H264Encoder
from picamera2.outputs import FfmpegOutput


def get_cfg(cfg_path: str):
    cfg = configparser.ConfigParser()
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    cfg.read(cfg_path)

    if "camera" not in cfg:
        raise ValueError("Config must contain a [camera] section")

    c = cfg["camera"]

    def get_int(key, default):
        return int(c.get(key, default))

    def get_float(key, default):
        return float(c.get(key, default))

    params = {
        "width": get_int("width", 640),
        "height": get_int("height", 480),
        "fps": get_int("fps", 60),

        "auto_exposure": get_int("auto_exposure", 0),
        "exposure_us": get_int("exposure_us", 8000),
        "analogue_gain": get_float("analogue_gain", 4.0),

        "disable_awb": get_int("disable_awb", 1),

        # If you want to hard-lock FPS:
        "lock_fps": get_int("lock_fps", 1),
    }
    return params


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="camera.cfg", help="Path to config file (.cfg/.ini)")
    ap.add_argument("--outdir", default="recordings", help="Root directory to save videos")
    ap.add_argument("--window", default="Preview (ESC=start/stop)", help="OpenCV window name")
    ap.add_argument("--bitrate", type=int, default=6_000_000, help="H264 bitrate (bps)")
    args = ap.parse_args()

    params = get_cfg(args.config)

    mouse_id = input("Mouse ID: ").strip()
    if not mouse_id:
        raise ValueError("Mouse ID cannot be empty.")

    # per-mouse folder
    mouse_dir = os.path.join(args.outdir, mouse_id)
    os.makedirs(mouse_dir, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    base = f"{mouse_id}_{timestamp}"
    out_mp4 = os.path.join(mouse_dir, f"{base}.mp4")
    out_csv = os.path.join(mouse_dir, f"{base}_timestamps.csv")

    picam2 = Picamera2()

    # Optional: hard-lock fps using frame duration limits
    controls_cfg = {"FrameRate": params["fps"]}
    if params["lock_fps"]:
        frame_us = int(round(1_000_000 / params["fps"]))  # e.g. 16667 for 60fps
        controls_cfg["FrameDurationLimits"] = (frame_us, frame_us)

    video_config = picam2.create_video_configuration(
        main={"size": (params["width"], params["height"]), "format": "RGB888"},
        controls=controls_cfg,
    )
    picam2.configure(video_config)

    # Manual exposure/gain + disable automatics
    controls = {}
    if params["auto_exposure"] == 1:
        controls["AeEnable"] = True
    else:
        controls["AeEnable"] = False
        controls["ExposureTime"] = int(params["exposure_us"])     # microseconds
        controls["AnalogueGain"] = float(params["analogue_gain"])

    if params["disable_awb"] == 1:
        controls["AwbEnable"] = False

    picam2.set_controls(controls)
    picam2.start()

    encoder = H264Encoder(bitrate=args.bitrate)
    output = FfmpegOutput(out_mp4)

    recording = False

    # Timestamp logging
    csv_f = None
    csv_w = None
    frame_idx = 0

    print("\nPreview running.")
    print("Press ESC to START recording, ESC again to STOP.\n")
    print(f"Will save video to: {out_mp4}")

    try:
        while True:
            # Use capture_request so we can also grab metadata timestamps
            req = picam2.capture_request()
            try:
                frame = req.make_array("main")  # RGB

                # Only log timestamps while recording
                if recording and csv_w is not None:
                    md = req.get_metadata() or {}

                    # Best-case: sensor timestamp (usually ns, monotonic)
                    sensor_ts_ns = md.get("SensorTimestamp", None)

                    # Also log your “human readable” time like your other script
                    wall_clock = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")

                    # And a monotonic clock in ns (good for sync)
                    mono_ns = time.monotonic_ns()

                    csv_w.writerow([
                        frame_idx,
                        wall_clock,
                        mono_ns,
                        sensor_ts_ns
                    ])
                    frame_idx += 1

            finally:
                req.release()

            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            status = "REC" if recording else "PREVIEW"
            cv2.putText(frame_bgr, status, (15, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        (0, 0, 255) if recording else (255, 255, 255), 2)
            cv2.imshow(args.window, frame_bgr)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                if not recording:
                    # open csv + write header
                    csv_f = open(out_csv, "w", newline="")
                    csv_w = csv.writer(csv_f)
                    csv_w.writerow(["frame_idx", "wall_clock", "monotonic_ns", "sensor_timestamp_ns"])

                    # start recording
                    picam2.start_recording(encoder, output)
                    recording = True
                    frame_idx = 0
                    print(f"Recording started -> {out_mp4}")
                    print(f"Timestamps -> {out_csv}")

                else:
                    picam2.stop_recording()
                    recording = False
                    print(f"Recording stopped -> {out_mp4}")

                    if csv_f is not None:
                        csv_f.flush()
                        csv_f.close()
                        csv_f = None
                        csv_w = None

                    break

    finally:
        try:
            if recording:
                picam2.stop_recording()
        except Exception:
            pass

        try:
            if csv_f is not None:
                csv_f.flush()
                csv_f.close()
        except Exception:
            pass

        picam2.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
