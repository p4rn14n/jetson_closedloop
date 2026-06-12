#!/usr/bin/env python3
"""
Raspberry Pi 5 camera GUI using Picamera2 with GPIO triggers + timestamp pre-callback,
AWB red/blue gains, and JSON save/load for full configuration.

Dependencies (Raspberry Pi OS Bookworm or later):
  sudo apt update
  sudo apt install -y python3-picamera2 python3-lgpio ffmpeg python3-tk python3-opencv

Run:
  python3 rpi5_cam_gui.py

Key behaviors:
- Uses pre_callback for on-frame timestamp overlay (set before any start()).
- When recording starts: always stops camera & closes preview (as requested), then starts recording.
- Still capture does not change preview state.
- Frame trigger uses post_callback property (set before recording) for minimal overhead, aligned with encoded frames.
- Dark-frame trigger: goes HIGH after N frames, and LOW N frames before end (if duration set);
  if open-ended, goes LOW upon stop.
- Save/Load config captures everything (camera, modes, sizes, controls, AWB gains, pins, paths, etc.).
"""

import os
import json
import threading
import time
from datetime import datetime
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import cv2

try:
    import lgpio as lg
except Exception as e:
    lg = None

try:
    from picamera2 import Picamera2, Preview, MappedArray
    from picamera2.encoders import H264Encoder
    from picamera2.outputs import FfmpegOutput
except Exception as e:
    raise SystemExit("Picamera2 not available: {}".format(e))


# ----------------------------- GPIO Helper -----------------------------
class GPIOManager:
    def __init__(self):
        self.chip = None
        self.claimed = set()
        if lg is not None:
            try:
                self.chip = lg.gpiochip_open(0)
            except Exception:
                self.chip = None

    def is_ready(self):
        return lg is not None and self.chip is not None

    def claim_output(self, line, default=0):
        if line is None:
            return
        if not self.is_ready():
            raise RuntimeError("GPIO not initialized. Install python3-lgpio and run with permissions.")
        if line in self.claimed:
            return
        lg.gpio_claim_output(self.chip, line, default)
        self.claimed.add(line)

    def write(self, line, level):
        if line is None or not self.is_ready():
            return
        lg.gpio_write(self.chip, line, 1 if level else 0)

    def free(self, line):
        if line is None or not self.is_ready():
            return
        try:
            lg.gpio_free(self.chip, line)
        finally:
            self.claimed.discard(line)

    def cleanup(self):
        if not self.is_ready():
            return
        for line in list(self.claimed):
            try:
                lg.gpio_free(self.chip, line)
            except Exception:
                pass
        self.claimed.clear()
        try:
            lg.gpiochip_close(self.chip)
        except Exception:
            pass
        self.chip = None


# ----------------------------- Camera App -----------------------------
class CameraApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Raspberry Pi 5 Camera GUI (Picamera2)")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        # State
        self.picam2 = None
        self.current_cam_index = None
        self.preview_on = False
        self.is_recording = False
        self.preview_window_name = "Camera Preview"
        self.preview_loop_job = None
        self._auto_apply_job = None
        self._suppress_auto_apply = False

        # Recording state
        self.rec_thread = None
        self.stop_rec_flag = threading.Event()
        self.frames_seen = 0
        self.first_frame_seen = False
        self.dark_high = False
        self.dark_frames_n = 0
        self.target_total_frames = None  # when duration provided
        self.current_fps = 30.0

        # GPIO setup
        self.gpio = GPIOManager()

        # GUI
        self.build_gui()

        # Automatically apply GUI changes during preview. Recording also always
        # re-reads the current GUI values immediately before it starts.
        self._setup_auto_apply_traces()

        # Initialize camera list and select the first
        self.populate_cameras()
        if self.cam_select['values']:
            self.cam_select.current(0)
            self.on_camera_change(None)

    # --------------------- GUI Construction ---------------------
    def build_gui(self):
        pad = dict(padx=6, pady=4)

        top = ttk.Frame(self.root)
        top.pack(fill=tk.X, **pad)

        # Camera selection & preview
        ttk.Label(top, text="Camera:").grid(row=0, column=0, sticky=tk.W)
        self.cam_select = ttk.Combobox(top, state="readonly", width=40)
        self.cam_select.grid(row=0, column=1, sticky=tk.W)
        self.cam_select.bind("<<ComboboxSelected>>", self.on_camera_change)

        self.preview_btn = ttk.Button(top, text="Start Preview", command=self.toggle_preview)
        self.preview_btn.grid(row=0, column=2, sticky=tk.W, padx=8)

        # Output path & prefix
        out = ttk.Frame(self.root)
        out.pack(fill=tk.X, **pad)

        ttk.Label(out, text="Output folder:").grid(row=0, column=0, sticky=tk.W)
        self.out_path_var = tk.StringVar(value=os.path.expanduser("~/Videos"))
        self.out_path_entry = ttk.Entry(out, textvariable=self.out_path_var, width=48)
        self.out_path_entry.grid(row=0, column=1, sticky=tk.W)
        ttk.Button(out, text="Browse", command=self.browse_folder).grid(row=0, column=2, sticky=tk.W)

        ttk.Label(out, text="File prefix:").grid(row=0, column=3, sticky=tk.W)
        self.prefix_var = tk.StringVar(value="capture")
        ttk.Entry(out, textvariable=self.prefix_var, width=16).grid(row=0, column=4, sticky=tk.W)

        # Sensor mode & image size
        cfg = ttk.LabelFrame(self.root, text="Capture Configuration")
        cfg.pack(fill=tk.X, **pad)

        ttk.Label(cfg, text="Sensor mode:").grid(row=0, column=0, sticky=tk.W)
        self.sensor_mode_cb = ttk.Combobox(cfg, state="readonly", width=60)
        self.sensor_mode_cb.grid(row=0, column=1, columnspan=5, sticky=tk.W)
        self.sensor_mode_cb.bind("<<ComboboxSelected>>", self.on_sensor_mode_select)

        ttk.Label(cfg, text="Width:").grid(row=1, column=0, sticky=tk.W)
        self.width_var = tk.IntVar(value=1920)
        self.width_entry = ttk.Entry(cfg, textvariable=self.width_var, width=8)
        self.width_entry.grid(row=1, column=1, sticky=tk.W)
        ttk.Label(cfg, text="Height:").grid(row=1, column=2, sticky=tk.W)
        self.height_var = tk.IntVar(value=1080)
        self.height_entry = ttk.Entry(cfg, textvariable=self.height_var, width=8)
        self.height_entry.grid(row=1, column=3, sticky=tk.W)
        ttk.Button(cfg, text="Apply Size / Mode", command=self.apply_size_mode).grid(row=1, column=4, sticky=tk.W, padx=8)

        # Controls: FPS, Shutter, Gain, AWB, ColourGains
        ctrl = ttk.LabelFrame(self.root, text="Controls")
        ctrl.pack(fill=tk.X, **pad)

        ttk.Label(ctrl, text="Frame rate (max 60):").grid(row=0, column=0, sticky=tk.W)
        self.fps_var = tk.DoubleVar(value=30.0)
        self.fps_entry = ttk.Entry(ctrl, textvariable=self.fps_var, width=8)
        self.fps_entry.grid(row=0, column=1, sticky=tk.W)

        ttk.Label(ctrl, text="Shutter (µs):").grid(row=0, column=2, sticky=tk.W)
        self.shutter_var = tk.IntVar(value=10000)
        self.shutter_entry = ttk.Entry(ctrl, textvariable=self.shutter_var, width=10)
        self.shutter_entry.grid(row=0, column=3, sticky=tk.W)

        ttk.Label(ctrl, text="Analogue Gain:").grid(row=0, column=4, sticky=tk.W)
        self.gain_var = tk.DoubleVar(value=1.0)
        self.gain_entry = ttk.Entry(ctrl, textvariable=self.gain_var, width=8)
        self.gain_entry.grid(row=0, column=5, sticky=tk.W)

        self.awb_available = False
        self.awb_var = tk.BooleanVar(value=True)
        self.awb_chk = ttk.Checkbutton(ctrl, text="AWB enabled", variable=self.awb_var, command=self.on_awb_toggle)
        self.awb_chk.grid(row=0, column=6, sticky=tk.W, padx=8)

        ttk.Label(ctrl, text="AWB Red Gain:").grid(row=1, column=0, sticky=tk.W)
        self.awb_r_var = tk.DoubleVar(value=1.8)
        self.awb_r_entry = ttk.Entry(ctrl, textvariable=self.awb_r_var, width=8)
        self.awb_r_entry.grid(row=1, column=1, sticky=tk.W)

        ttk.Label(ctrl, text="AWB Blue Gain:").grid(row=1, column=2, sticky=tk.W)
        self.awb_b_var = tk.DoubleVar(value=1.5)
        self.awb_b_entry = ttk.Entry(ctrl, textvariable=self.awb_b_var, width=8)
        self.awb_b_entry.grid(row=1, column=3, sticky=tk.W)

        # Timestamp overlay
        self.ts_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(ctrl, text="Overlay timestamp", variable=self.ts_var).grid(row=1, column=4, sticky=tk.W, padx=8)

        ttk.Button(ctrl, text="Apply Controls", command=self.apply_controls).grid(row=1, column=6, sticky=tk.W, padx=8)

        # Triggers
        trig = ttk.LabelFrame(self.root, text="GPIO Triggers (BCM numbering)")
        trig.pack(fill=tk.X, **pad)

        pins = ["None"] + [str(p) for p in range(2, 28)]

        # Recording trigger
        ttk.Label(trig, text="Recording trigger pin:").grid(row=0, column=0, sticky=tk.W)
        self.rec_pin_var = tk.StringVar(value="None")
        self.rec_pin_cb = ttk.Combobox(trig, values=pins, state="readonly", width=6, textvariable=self.rec_pin_var)
        self.rec_pin_cb.grid(row=0, column=1, sticky=tk.W)

        # Frame trigger
        ttk.Label(trig, text="Frame trigger pin:").grid(row=0, column=2, sticky=tk.W)
        self.frame_pin_var = tk.StringVar(value="None")
        self.frame_pin_cb = ttk.Combobox(trig, values=pins, state="readonly", width=6, textvariable=self.frame_pin_var)
        self.frame_pin_cb.grid(row=0, column=3, sticky=tk.W)

        ttk.Label(trig, text="Pulse width (ms):").grid(row=0, column=4, sticky=tk.W)
        self.pulse_ms_var = tk.DoubleVar(value=2.0)
        ttk.Entry(trig, textvariable=self.pulse_ms_var, width=6).grid(row=0, column=5, sticky=tk.W)

        # Dark-frame trigger
        ttk.Label(trig, text="Dark trigger pin:").grid(row=1, column=0, sticky=tk.W)
        self.dark_pin_var = tk.StringVar(value="None")
        self.dark_pin_cb = ttk.Combobox(trig, values=pins, state="readonly", width=6, textvariable=self.dark_pin_var)
        self.dark_pin_cb.grid(row=1, column=1, sticky=tk.W)

        ttk.Label(trig, text="Dark frames N:").grid(row=1, column=2, sticky=tk.W)
        self.dark_n_var = tk.IntVar(value=0)
        ttk.Entry(trig, textvariable=self.dark_n_var, width=6).grid(row=1, column=3, sticky=tk.W)

        ttk.Label(trig, text="(HIGH after N frames; LOW N frames before end if duration set)").grid(row=1, column=4, columnspan=3, sticky=tk.W)

        # Capture controls
        cap = ttk.LabelFrame(self.root, text="Capture")
        cap.pack(fill=tk.X, **pad)

        ttk.Button(cap, text="Capture STILL", command=self.capture_still).grid(row=0, column=0, sticky=tk.W)

        ttk.Label(cap, text="Video duration (sec, blank = until stop):").grid(row=0, column=1, sticky=tk.W)
        self.duration_var = tk.StringVar(value="")
        ttk.Entry(cap, textvariable=self.duration_var, width=8).grid(row=0, column=2, sticky=tk.W)

        self.rec_btn = ttk.Button(cap, text="Start RECORD", command=self.start_recording)
        self.rec_btn.grid(row=0, column=3, sticky=tk.W, padx=6)
        self.stop_btn = ttk.Button(cap, text="Stop", command=self.stop_recording)
        self.stop_btn.grid(row=0, column=4, sticky=tk.W)

        # Config save/load
        conf = ttk.Frame(self.root)
        conf.pack(fill=tk.X, **pad)
        ttk.Button(conf, text="Save Config", command=self.save_config).grid(row=0, column=0, sticky=tk.W)
        ttk.Button(conf, text="Load Config", command=self.load_config).grid(row=0, column=1, sticky=tk.W, padx=6)

        # Status bar
        self.status_var = tk.StringVar(value="Ready.")
        ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W).pack(fill=tk.X, side=tk.BOTTOM)

    # --------------------- Camera Discovery ---------------------
    def populate_cameras(self):
        try:
            info = Picamera2.global_camera_info()
        except Exception:
            info = []
        items = []
        for i, cam in enumerate(info):
            model = cam.get('Model', cam.get('Id', f'Camera {i}'))
            items.append(f"{i}: {model}")
        self.cam_select['values'] = items
        if not items:
            messagebox.showerror("No cameras", "No cameras detected by Picamera2/libcamera.")

    def on_camera_change(self, event):
        sel = self.cam_select.get()
        if not sel:
            return
        idx = int(sel.split(":")[0])

        # Avoid re-opening the same camera. On Raspberry Pi this can raise
        # "Device or resource busy" if the old Picamera2 object is still active.
        if idx == self.current_cam_index and self.picam2 is not None:
            return

        if self.is_recording:
            messagebox.showwarning("Busy", "Stop recording before switching cameras.")
            self._set_camera_combo_to_current()
            return

        self.switch_camera(idx)

    def switch_camera(self, index):
        # Stop and release the existing Picamera2 object before opening another one.
        old_index = self.current_cam_index
        try:
            if self.picam2 is not None:
                if self.is_recording:
                    self.stop_recording()
                self._stop_preview(show_status=False)
                try:
                    self.picam2.stop()
                except Exception:
                    pass
                try:
                    self.picam2.stop_preview()
                except Exception:
                    pass
                try:
                    self.picam2.close()
                except Exception:
                    pass
        except Exception:
            pass

        self.preview_on = False
        self.is_recording = False
        try:
            self.picam2 = Picamera2(index)
        except Exception as e:
            self.current_cam_index = old_index
            self._set_camera_combo_to_current()
            messagebox.showerror("Camera error", str(e))
            return
        self.current_cam_index = index

        # Default configuration
        try:
            cfg = self.picam2.create_video_configuration(main={"size": (1920, 1080)})
            self.picam2.configure(cfg)
        except Exception:
            pass

        # Populate sensor modes
        modes = []
        try:
            modes = self.picam2.sensor_modes
        except Exception:
            modes = []
        display = []
        for m in modes:
            sz = m.get('size', (0, 0))
            fps = m.get('fps', 0)
            bit = m.get('bit_depth', '')
            fmt = m.get('format', '')
            display.append(f"{sz[0]}x{sz[1]} @ {fps}fps {bit}bit {fmt}")
        self.sensor_mode_cb['values'] = display
        if display:
            self.sensor_mode_cb.current(0)
            first = modes[0]
            w, h = first.get('size', (1920, 1080))
            self.width_var.set(w)
            self.height_var.set(h)
            self.fps_var.set(min(60.0, float(first.get('fps', 30))))

        # AWB control availability
        self.awb_available = self._has_control('AwbEnable') or self._has_control('ColourGains')
        if self.awb_available:
            self.awb_chk.state(["!disabled"])  # enable
        else:
            self.awb_chk.state(["disabled"])   # disable
        self.on_awb_toggle()  # set gains fields enabled/disabled

        self.status("Camera switched.")


    def _set_camera_combo_to_current(self):
        if self.current_cam_index is None:
            return
        try:
            for item in self.cam_select['values']:
                if item.startswith(f"{self.current_cam_index}:"):
                    self.cam_select.set(item)
                    break
        except Exception:
            pass

    def _setup_auto_apply_traces(self):
        # Size/mode changes need a camera reconfigure; controls can be applied live.
        for var in (self.width_var, self.height_var):
            var.trace_add("write", lambda *_: self._schedule_auto_apply(kind="size"))

        for var in (self.fps_var, self.shutter_var, self.gain_var,
                    self.awb_var, self.awb_r_var, self.awb_b_var, self.ts_var):
            var.trace_add("write", lambda *_: self._schedule_auto_apply(kind="controls"))

    def _schedule_auto_apply(self, kind="controls"):
        if self._suppress_auto_apply or self.is_recording or self.picam2 is None:
            return

        # During preview, apply changes automatically after the user pauses typing.
        # Outside preview, start_recording() still applies the latest GUI values.
        if not self.preview_on:
            return

        if self._auto_apply_job is not None:
            try:
                self.root.after_cancel(self._auto_apply_job)
            except Exception:
                pass
        delay = 700 if kind == "size" else 250
        self._auto_apply_job = self.root.after(delay, lambda: self._auto_apply_now(kind))

    def _auto_apply_now(self, kind="controls"):
        self._auto_apply_job = None
        if self.is_recording or self.picam2 is None or not self.preview_on:
            return
        if kind == "size":
            self._restart_preview()
        else:
            self._set_pre_callback()
            self.apply_controls(show_errors=False)

    def _has_control(self, name: str) -> bool:
        try:
            cc = self.picam2.camera_controls
            return name in cc
        except Exception:
            return False

    # --------------------- Config / Controls ---------------------
    def on_sensor_mode_select(self, event):
        try:
            modes = self.picam2.sensor_modes
            i = self.sensor_mode_cb.current()
            if 0 <= i < len(modes):
                m = modes[i]
                w, h = m.get('size', (self.width_var.get(), self.height_var.get()))
                self.width_var.set(w)
                self.height_var.set(h)
                fps = float(m.get('fps', self.fps_var.get()))
                self.fps_var.set(min(60.0, fps))
                self.status("Sensor mode selected; size/FPS updated.")
                self._schedule_auto_apply(kind="size")
        except Exception:
            pass

    def apply_size_mode(self, show_errors=True):
        if self.is_recording:
            if show_errors:
                messagebox.showwarning("Busy", "Stop recording before changing size/mode.")
            return
        w = int(self.width_var.get())
        h = int(self.height_var.get())
        if w <= 0 or h <= 0:
            if show_errors:
                messagebox.showerror("Invalid size", "Width and height must be positive.")
            return
        try:
            # Stop camera if running to reconfigure
            try:
                self.picam2.stop()
            except Exception:
                pass
            cfg = self._video_config()
            self.picam2.configure(cfg)
            self.status(f"Applied size {w}x{h}.")
        except Exception as e:
            if show_errors:
                messagebox.showerror("Configure error", str(e))

    def on_awb_toggle(self):
        # Enable ColourGains only when AWB disabled
        state = "!disabled" if not self.awb_var.get() else "disabled"
        try:
            self.awb_r_entry.state([state])
            self.awb_b_entry.state([state])
        except Exception:
            # Fallback for older ttk
            self.awb_r_entry.config(state=("normal" if not self.awb_var.get() else "disabled"))
            self.awb_b_entry.config(state=("normal" if not self.awb_var.get() else "disabled"))

    def apply_controls(self, show_errors=True):
        fps = max(1.0, min(60.0, float(self.fps_var.get())))
        self.current_fps = fps
        controls = {
            "FrameRate": float(fps),
            "ExposureTime": int(max(1, self.shutter_var.get())),
            "AnalogueGain": float(max(1.0, self.gain_var.get())),
        }
        if self.awb_available:
            controls["AwbEnable"] = bool(self.awb_var.get())
            if not self.awb_var.get():
                # Manual colour gains when AWB disabled
                try:
                    rg = float(self.awb_r_var.get())
                    bg = float(self.awb_b_var.get())
                    controls["ColourGains"] = (rg, bg)
                except Exception:
                    pass
        try:
            self.picam2.set_controls(controls)
            self.status("Controls applied.")
        except Exception as e:
            if show_errors:
                messagebox.showerror("Control error", str(e))

    def _set_pre_callback(self):
        # Set or clear pre_callback before any start()
        if not self.ts_var.get():
            self.picam2.pre_callback = None
            return

        def apply_timestamp(request):
            try:
                ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                with MappedArray(request, "main") as m:
                    img = m.array
                    h = img.shape[0]
                    org = (20, h - 20)
                    cv2.putText(img, ts, org, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            except Exception:
                pass

        self.picam2.pre_callback = apply_timestamp

    # --------------------- Preview ---------------------
    def toggle_preview(self):
        if self.is_recording:
            return  # disabled during record
        if self.preview_on:
            self._stop_preview()
        else:
            self._start_preview()

    def _start_preview(self):
        if self.picam2 is None or self.is_recording:
            return
        try:
            try:
                self.picam2.stop()
            except Exception:
                pass

            self.picam2.configure(self._video_config())
            self._set_pre_callback()
            self.apply_controls(show_errors=False)
            self.picam2.start()

            cv2.namedWindow(self.preview_window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.preview_window_name, 640, 480)
            self.preview_on = True
            self.preview_btn.config(text="Stop Preview")
            self.status("Preview started.")
            self._preview_loop()
        except Exception as e:
            self.preview_on = False
            self.preview_btn.config(text="Start Preview")
            try:
                cv2.destroyWindow(self.preview_window_name)
            except Exception:
                pass
            messagebox.showerror("Preview error", str(e))

    def _preview_loop(self):
        if not self.preview_on or self.picam2 is None:
            return

        # If the user clicks X on the OpenCV preview window, make it equivalent
        # to pressing the GUI "Stop Preview" button.
        try:
            if cv2.getWindowProperty(self.preview_window_name, cv2.WND_PROP_VISIBLE) < 1:
                self._stop_preview()
                return
        except Exception:
            self._stop_preview()
            return

        try:
            frame = self.picam2.capture_array("main")
            # Picamera2 usually gives RGB/RGBA arrays; OpenCV displays BGR/BGRA.
            if frame.ndim == 3 and frame.shape[2] == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            elif frame.ndim == 3 and frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGRA)

            # Display only; recorded frames are NOT resized by this preview.
            h, w = frame.shape[:2]
            max_w, max_h = 640, 480
            scale = min(max_w / max(1, w), max_h / max(1, h), 1.0)
            if scale < 1.0:
                frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

            cv2.imshow(self.preview_window_name, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')):
                self._stop_preview()
                return
        except Exception as e:
            self._stop_preview(show_status=False)
            messagebox.showerror("Preview error", str(e))
            return

        self.preview_loop_job = self.root.after(15, self._preview_loop)

    def _stop_preview(self, show_status=True):
        if self._auto_apply_job is not None:
            try:
                self.root.after_cancel(self._auto_apply_job)
            except Exception:
                pass
            self._auto_apply_job = None
        if self.preview_loop_job is not None:
            try:
                self.root.after_cancel(self.preview_loop_job)
            except Exception:
                pass
            self.preview_loop_job = None
        try:
            cv2.destroyWindow(self.preview_window_name)
            cv2.waitKey(1)
        except Exception:
            pass
        try:
            if self.picam2 is not None:
                self.picam2.stop()
        except Exception:
            pass
        try:
            if self.picam2 is not None:
                self.picam2.stop_preview()
        except Exception:
            pass
        self.preview_on = False
        self.preview_btn.config(text="Start Preview")
        if show_status:
            self.status("Preview stopped.")

    def _restart_preview(self):
        if not self.preview_on:
            return
        self._stop_preview(show_status=False)
        self._start_preview()

    # --------------------- File helpers ---------------------
    def browse_folder(self):
        d = filedialog.askdirectory(initialdir=self.out_path_var.get() or os.path.expanduser("~"))
        if d:
            self.out_path_var.set(d)

    def _ensure_outdir(self):
        path = self.out_path_var.get().strip()
        if not path:
            raise RuntimeError("Output folder is empty.")
        os.makedirs(path, exist_ok=True)
        return path

    def _timestamped(self, ext: str):
        prefix = self.prefix_var.get().strip() or "capture"
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{prefix}_{ts}.{ext}"

    # --------------------- Still ---------------------
    def capture_still(self):
        if self.is_recording:
            messagebox.showwarning("Busy", "Stop recording before taking a still.")
            return
        try:
            outdir = self._ensure_outdir()
            fname = os.path.join(outdir, self._timestamped("png"))
            # Ensure pre-callback is set (timestamp overlay)
            self._set_pre_callback()

            # Use still configuration at requested size regardless of preview state
            try:
                self.picam2.switch_mode_and_capture_file(self._still_config(), fname)
            except Exception:
                # Fallback path
                try:
                    self.picam2.stop()
                except Exception:
                    pass
                self.picam2.configure(self._still_config())
                self.picam2.start()
                self.picam2.capture_file(fname)
                self.picam2.stop()

            self.status(f"Saved still: {fname}")
        except Exception as e:
            messagebox.showerror("Capture error", str(e))

    # --------------------- Recording ---------------------
    def _parse_duration(self):
        txt = self.duration_var.get().strip()
        if not txt:
            return None
        try:
            val = float(txt)
            if val <= 0:
                return None
            return val
        except Exception:
            return None

    def _get_pin(self, var):
        s = var.get()
        return None if s in (None, "", "None") else int(s)

    def _check_pin_conflicts(self, pins):
        pins_no_none = [p for p in pins if p is not None]
        return len(pins_no_none) == len(set(pins_no_none))

    def start_recording(self):
        if self.is_recording:
            return
        try:
            outdir = self._ensure_outdir()
        except Exception as e:
            messagebox.showerror("Output", str(e))
            return

        rec_pin = self._get_pin(self.rec_pin_var)
        frame_pin = self._get_pin(self.frame_pin_var)
        dark_pin = self._get_pin(self.dark_pin_var)
        pins = [rec_pin, frame_pin, dark_pin]
        if not self._check_pin_conflicts(pins):
            messagebox.showerror("GPIO conflict", "The same GPIO pin is assigned to multiple trigger types.")
            return

        # Claim configured pins
        try:
            for p in pins:
                if p is not None:
                    self.gpio.claim_output(p, 0)
        except Exception as e:
            messagebox.showerror("GPIO error", str(e))
            return

        # Prepare recording
        duration = self._parse_duration()  # seconds or None
        self.current_fps = max(1.0, min(60.0, float(self.fps_var.get())))
        if duration is not None:
            self.target_total_frames = int(round(self.current_fps * duration))
        else:
            self.target_total_frames = None

        self.dark_frames_n = max(0, int(self.dark_n_var.get()))
        self.frames_seen = 0
        self.first_frame_seen = False
        self.dark_high = False
        self.stop_rec_flag.clear()

        # Prepare encoder/output
        video_name = os.path.join(outdir, self._timestamped("h264"))
        encoder = H264Encoder()
        output = video_name #FfmpegOutput(video_name)

        # Post-callback for per-frame logic (SET AS PROPERTY BEFORE START)
        def post_cb(*_):
            # Called after each encoded frame
            self.frames_seen += 1

            # First-frame: set recording line HIGH here for accuracy
            if not self.first_frame_seen:
                if rec_pin is not None:
                    self.gpio.write(rec_pin, 1)
                self.first_frame_seen = True

            # Frame trigger pulse
            if frame_pin is not None:
                self._pulse(frame_pin, self.pulse_ms_var.get())

            # Dark-frame trigger logic
            if dark_pin is not None and self.dark_frames_n > 0:
                if not self.dark_high and self.frames_seen >= self.dark_frames_n:
                    self.gpio.write(dark_pin, 1)
                    self.dark_high = True
                if self.target_total_frames is not None:
                    # planned-duration mode
                    if self.dark_high and self.frames_seen >= max(0, self.target_total_frames - self.dark_frames_n):
                        self.gpio.write(dark_pin, 0)

        # Stop preview/camera and configure from the current GUI values.
        # This prevents the common mistake where values were typed but not applied.
        self._stop_preview(show_status=False)
        try:
            self.picam2.stop()
        except Exception:
            pass
        try:
            self.picam2.configure(self._video_config())
        except Exception as e:
            messagebox.showerror("Configure error", str(e))
            return

        # Ensure pre- and post-callbacks set BEFORE starting
        self._set_pre_callback()
        self.picam2.post_callback = post_cb

        # Apply controls just before recording
        self.apply_controls()

        # Start the recording (Picamera2 will start the camera internally)
        try:
            self.picam2.start_recording(encoder, output, name="main", pts=video_name[:-4]+'txt')
        except Exception as e:
            messagebox.showerror("Record error", str(e))
            for p in (rec_pin, frame_pin, dark_pin):
                try:
                    if p is not None:
                        self.gpio.write(p, 0)
                        self.gpio.free(p)
                except Exception:
                    pass
            return

        # Disable preview toggle while recording
        self.preview_btn.state(["disabled"])  # cannot change during record

        self.is_recording = True
        self.status(f"Recording to {video_name}")

        # Start thread to stop after duration, if provided
        if duration is not None:
            self.rec_thread = threading.Thread(target=self._auto_stop_after, args=(duration,), daemon=True)
            self.rec_thread.start()

    def _auto_stop_after(self, seconds):
        # Sleep without blocking GUI; allow manual stop earlier
        t_end = time.time() + seconds
        while not self.stop_rec_flag.is_set() and time.time() < t_end:
            time.sleep(0.05)
        if not self.stop_rec_flag.is_set():
            self.root.after(0, self.stop_recording)

    def _pulse(self, pin: int, ms: float):
        if pin is None:
            return
        try:
            self.gpio.write(pin, 1)
            # Use a short timer to drop the line back low
            threading.Timer(max(0.001, ms / 1000.0), lambda: self.gpio.write(pin, 0)).start()
        except Exception:
            pass

    def stop_recording(self):
        if not self.is_recording:
            return
        self.stop_rec_flag.set()
        try:
            self.picam2.stop_recording()
        except Exception:
            pass

        # Drop GPIO lines LOW at end
        rec_pin = self._get_pin(self.rec_pin_var)
        frame_pin = self._get_pin(self.frame_pin_var)
        dark_pin = self._get_pin(self.dark_pin_var)
        for p in (rec_pin, frame_pin, dark_pin):
            try:
                if p is not None:
                    self.gpio.write(p, 0)
            except Exception:
                pass
        # Free lines
        for p in (rec_pin, frame_pin, dark_pin):
            try:
                if p is not None:
                    self.gpio.free(p)
            except Exception:
                pass

        try:
            self.picam2.post_callback = None
        except Exception:
            pass
        try:
            # Release the camera after recording so preview can be started again
            # without "Device or resource busy".
            self.picam2.stop()
        except Exception:
            pass

        self.is_recording = False
        self.preview_btn.state(["!disabled"])  # re-enable
        self.status("Recording stopped.")

    # --------------------- Config Save/Load ---------------------
    def _current_config_dict(self):
        return {
            "camera_index": self.current_cam_index,
            "sensor_mode_index": self.sensor_mode_cb.current(),
            "width": int(self.width_var.get()),
            "height": int(self.height_var.get()),
            "fps": float(self.fps_var.get()),
            "shutter_us": int(self.shutter_var.get()),
            "analogue_gain": float(self.gain_var.get()),
            "awb_enable": bool(self.awb_var.get()),
            "awb_red_gain": float(self.awb_r_var.get()),
            "awb_blue_gain": float(self.awb_b_var.get()),
            "overlay_timestamp": bool(self.ts_var.get()),
            "output_folder": self.out_path_var.get(),
            "file_prefix": self.prefix_var.get(),
            "duration": self.duration_var.get(),
            "pins": {
                "record": self.rec_pin_var.get(),
                "frame": self.frame_pin_var.get(),
                "dark": self.dark_pin_var.get(),
                "pulse_ms": float(self.pulse_ms_var.get()),
                "dark_n": int(self.dark_n_var.get()),
            },
        }

    def _apply_config_dict(self, cfg):
        self._suppress_auto_apply = True
        try:
            self._apply_config_dict_inner(cfg)
        finally:
            self._suppress_auto_apply = False

    def _apply_config_dict_inner(self, cfg):
        # Camera first
        cam_idx = cfg.get("camera_index")
        if cam_idx is not None:
            # Validate available cameras
            items = self.cam_select['values']
            indices = [int(it.split(":")[0]) for it in items]
            if cam_idx in indices:
                self.cam_select.set(next(it for it in items if it.startswith(f"{cam_idx}:")))
                self.switch_camera(cam_idx)
            else:
                messagebox.showwarning("Camera", "Saved camera index not available on this system.")

        # Simple fields
        self.width_var.set(int(cfg.get("width", self.width_var.get())))
        self.height_var.set(int(cfg.get("height", self.height_var.get())))
        self.fps_var.set(float(cfg.get("fps", self.fps_var.get())))
        self.shutter_var.set(int(cfg.get("shutter_us", self.shutter_var.get())))
        self.gain_var.set(float(cfg.get("analogue_gain", self.gain_var.get())))

        self.awb_var.set(bool(cfg.get("awb_enable", self.awb_var.get())))
        self.on_awb_toggle()
        self.awb_r_var.set(float(cfg.get("awb_red_gain", self.awb_r_var.get())))
        self.awb_b_var.set(float(cfg.get("awb_blue_gain", self.awb_b_var.get())))

        self.ts_var.set(bool(cfg.get("overlay_timestamp", self.ts_var.get())))

        self.out_path_var.set(cfg.get("output_folder", self.out_path_var.get()))
        self.prefix_var.set(cfg.get("file_prefix", self.prefix_var.get()))
        self.duration_var.set(cfg.get("duration", self.duration_var.get()))

        pins = cfg.get("pins", {})
        self.rec_pin_var.set(pins.get("record", self.rec_pin_var.get()))
        self.frame_pin_var.set(pins.get("frame", self.frame_pin_var.get()))
        self.dark_pin_var.set(pins.get("dark", self.dark_pin_var.get()))
        self.pulse_ms_var.set(float(pins.get("pulse_ms", self.pulse_ms_var.get())))
        self.dark_n_var.set(int(pins.get("dark_n", self.dark_n_var.get())))

        # Sensor mode index (after camera switch so modes are populated)
        smi = cfg.get("sensor_mode_index")
        if smi is not None and 0 <= smi < len(self.sensor_mode_cb['values']):
            self.sensor_mode_cb.current(smi)

        # Apply size/mode to camera
        self.apply_size_mode()
        # Apply controls to camera
        self.apply_controls()

    def save_config(self):
        cfg = self._current_config_dict()
        fpath = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON", "*.json")], initialfile="rpi_cam_config.json")
        if not fpath:
            return
        try:
            with open(fpath, 'w') as f:
                json.dump(cfg, f, indent=2)
            self.status(f"Saved config: {fpath}")
        except Exception as e:
            messagebox.showerror("Save Config", str(e))

    def load_config(self):
        fpath = filedialog.askopenfilename(filetypes=[("JSON", "*.json")])
        if not fpath:
            return
        try:
            with open(fpath, 'r') as f:
                cfg = json.load(f)
            self._apply_config_dict(cfg)
            self.status(f"Loaded config: {fpath}")
        except Exception as e:
            messagebox.showerror("Load Config", str(e))

    # --------------------- Helper Configs ---------------------
    def _video_config(self):
        w = int(self.width_var.get())
        h = int(self.height_var.get())
        # Keep RGB main so timestamp overlay is simple; encoder will convert internally
        return self.picam2.create_video_configuration(main={"size": (w, h)})

    def _still_config(self):
        w = int(self.width_var.get())
        h = int(self.height_var.get())
        return self.picam2.create_still_configuration(main={"size": (w, h)})

    # --------------------- Utils ---------------------
    def status(self, msg: str):
        self.status_var.set(msg)
        self.root.update_idletasks()

    def on_close(self):
        try:
            if self.is_recording:
                self.stop_recording()
        except Exception:
            pass
        try:
            self._stop_preview(show_status=False)
        except Exception:
            pass
        try:
            if self.picam2 is not None:
                self.picam2.close()
        except Exception:
            pass
        try:
            self.gpio.cleanup()
        except Exception:
            pass
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    style = ttk.Style(root)
    try:
        style.theme_use('clam')
    except Exception:
        pass
    app = CameraApp(root)
    root.mainloop()
