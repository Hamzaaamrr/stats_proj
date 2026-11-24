"""Tk dashboard for the live driver monitoring session."""
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
from PIL import Image, ImageTk


BG = "#0b1220"
SURFACE = "#111c2e"
SURFACE_ALT = "#17243a"
TEXT = "#e7edf7"
MUTED = "#91a0b7"
ACCENT = "#35c69a"
RED = "#f36b76"


class DriverMonitorApp:
    def __init__(self, cfg, args, worker_factory):
        self.cfg = cfg
        self.args = args
        self.worker_factory = worker_factory
        self.root = tk.Tk()
        self.root.title("Driver Monitor")
        self.root.geometry("1260x820")
        self.root.minsize(1060, 720)
        self.root.configure(bg=BG)
        self.root.protocol("WM_DELETE_WINDOW", self._close)
        self.root.bind("<Escape>", lambda _event: self._stop())
        self.root.bind("q", lambda event: self._hotkey(event, "q"))
        self.root.bind("c", lambda event: self._hotkey(event, "c"))
        self.root.bind("m", lambda event: self._hotkey(event, "m"))
        for key in "012345":
            self.root.bind(key, lambda event, value=key: self._hotkey(event, value))

        self.worker = None
        self.closing = False
        self.last_frame = None
        self.settings_widgets = []
        self.alerts_enabled = tk.BooleanVar(value=not args.no_audio)
        self.test_mode = tk.BooleanVar(value=args.test_mode)
        self.object_detection = tk.BooleanVar(value=not args.no_objects)

        self._configure_styles()
        self._build()
        self.root.after(40, self._poll)

    def run(self):
        self.root.mainloop()

    def _configure_styles(self):
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure("App.TFrame", background=BG)
        style.configure("Card.TFrame", background=SURFACE)
        style.configure("Metric.TFrame", background=SURFACE_ALT)
        style.configure("Title.TLabel", background=BG, foreground=TEXT,
                        font=("Segoe UI", 20, "bold"))
        style.configure("Subtitle.TLabel", background=BG, foreground=MUTED,
                        font=("Segoe UI", 10))
        style.configure("CardTitle.TLabel", background=SURFACE, foreground=MUTED,
                        font=("Segoe UI", 9, "bold"))
        style.configure("Body.TLabel", background=SURFACE, foreground=TEXT,
                        font=("Segoe UI", 10))
        style.configure("Muted.TLabel", background=SURFACE, foreground=MUTED,
                        font=("Segoe UI", 9))
        style.configure("MetricTitle.TLabel", background=SURFACE_ALT, foreground=MUTED,
                        font=("Segoe UI", 9, "bold"))
        style.configure("MetricValue.TLabel", background=SURFACE_ALT, foreground=TEXT,
                        font=("Segoe UI", 18, "bold"))
        style.configure("MetricNote.TLabel", background=SURFACE_ALT, foreground=MUTED,
                        font=("Segoe UI", 8))
        style.configure("State.TLabel", background=SURFACE, foreground=ACCENT,
                        font=("Segoe UI", 24, "bold"))
        style.configure("TEntry", fieldbackground="#0e1727", foreground=TEXT,
                        insertcolor=TEXT, bordercolor="#2b3b53", padding=7)
        style.configure("TCombobox", fieldbackground="#0e1727", foreground=TEXT,
                        arrowcolor=TEXT, padding=6)
        style.map("TCombobox", fieldbackground=[("readonly", "#0e1727")],
                  foreground=[("readonly", TEXT)])
        style.configure("TCheckbutton", background=SURFACE, foreground=TEXT,
                        font=("Segoe UI", 9))
        style.map("TCheckbutton", background=[("active", SURFACE)])
        style.configure("Accent.TButton", background=ACCENT, foreground="#07140f",
                        font=("Segoe UI", 10, "bold"), padding=(14, 9), borderwidth=0)
        style.map("Accent.TButton", background=[("active", "#57d8b0"), ("disabled", "#526678")])
        style.configure("Secondary.TButton", background="#253650", foreground=TEXT,
                        font=("Segoe UI", 9, "bold"), padding=(11, 8), borderwidth=0)
        style.map("Secondary.TButton", background=[("active", "#334966"), ("disabled", "#202d40")])
        style.configure("Danger.TButton", background="#6f2938", foreground="#ffe7eb",
                        font=("Segoe UI", 9, "bold"), padding=(11, 8), borderwidth=0)
        style.map("Danger.TButton", background=[("active", "#913746"), ("disabled", "#3b3440")])

    def _build(self):
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)

        header = ttk.Frame(self.root, style="App.TFrame", padding=(24, 18, 24, 14))
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(0, weight=1)
        ttk.Label(header, text="Driver Monitor", style="Title.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(header, text="Real-time driver safety dashboard",
                  style="Subtitle.TLabel").grid(row=1, column=0, sticky="w", pady=(3, 0))
        self.connection_label = tk.Label(
            header, text="●  READY", bg=BG, fg=ACCENT, font=("Segoe UI", 10, "bold"))
        self.connection_label.grid(row=0, column=1, rowspan=2, sticky="e")

        body = ttk.Frame(self.root, style="App.TFrame", padding=(24, 0, 24, 16))
        body.grid(row=1, column=0, sticky="nsew")
        body.columnconfigure(0, weight=3, uniform="body")
        body.columnconfigure(1, weight=2, uniform="body")
        body.rowconfigure(0, weight=1)

        self._build_preview(body)
        self._build_sidebar(body)

        footer = tk.Label(
            self.root,
            text="Educational prototype — not a certified safety system",
            bg=BG, fg="#66758c", font=("Segoe UI", 8),
        )
        footer.grid(row=2, column=0, sticky="w", padx=26, pady=(0, 12))

    def _build_preview(self, parent):
        panel = ttk.Frame(parent, style="Card.TFrame", padding=16)
        panel.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
        panel.columnconfigure(0, weight=1)
        panel.rowconfigure(1, weight=1)

        head = ttk.Frame(panel, style="Card.TFrame")
        head.grid(row=0, column=0, sticky="ew", pady=(0, 12))
        head.columnconfigure(0, weight=1)
        ttk.Label(head, text="LIVE CAMERA", style="CardTitle.TLabel").grid(row=0, column=0, sticky="w")
        self.session_label = ttk.Label(head, text="No active session", style="Muted.TLabel")
        self.session_label.grid(row=0, column=1, sticky="e")

        self.preview = tk.Label(
            panel, text="Ready to monitor\n\nStart a session to view the live camera.",
            bg="#050a12", fg="#8291a8", font=("Segoe UI", 12), justify="center",
        )
        self.preview.grid(row=1, column=0, sticky="nsew")

        controls = ttk.Frame(panel, style="Card.TFrame")
        controls.grid(row=2, column=0, sticky="ew", pady=(14, 0))
        for column in range(4):
            controls.columnconfigure(column, weight=1)
        self.start_button = ttk.Button(
            controls, text="Start monitoring", style="Accent.TButton", command=self._start)
        self.start_button.grid(row=0, column=0, sticky="ew", padx=(0, 6))
        self.stop_button = ttk.Button(
            controls, text="Stop", style="Danger.TButton", command=self._stop, state="disabled")
        self.stop_button.grid(row=0, column=1, sticky="ew", padx=6)
        self.recalibrate_button = ttk.Button(
            controls, text="Recalibrate", style="Secondary.TButton",
            command=lambda: self._command(ord("c")), state="disabled")
        self.recalibrate_button.grid(row=0, column=2, sticky="ew", padx=6)
        self.audio_button = ttk.Button(
            controls, text=self._audio_button_text(), style="Secondary.TButton",
            command=self._toggle_audio, state="disabled")
        self.audio_button.grid(row=0, column=3, sticky="ew", padx=(6, 0))

        self.message_label = ttk.Label(panel, text="Configure a session, then start monitoring.",
                                       style="Muted.TLabel")
        self.message_label.grid(row=3, column=0, sticky="w", pady=(12, 0))

    def _build_sidebar(self, parent):
        sidebar = ttk.Frame(parent, style="App.TFrame")
        sidebar.grid(row=0, column=1, sticky="nsew")
        sidebar.columnconfigure(0, weight=1)

        status = ttk.Frame(sidebar, style="Card.TFrame", padding=16)
        status.grid(row=0, column=0, sticky="ew", pady=(0, 12))
        status.columnconfigure(0, weight=1)
        ttk.Label(status, text="CURRENT BEHAVIOR", style="CardTitle.TLabel").grid(row=0, column=0, sticky="w")
        self.state_label = ttk.Label(status, text="Ready", style="State.TLabel")
        self.state_label.grid(row=1, column=0, sticky="w", pady=(7, 0))
        self.confidence_label = ttk.Label(status, text="No session running",
                                          style="Muted.TLabel")
        self.confidence_label.grid(row=2, column=0, sticky="w", pady=(2, 0))

        metrics = ttk.Frame(sidebar, style="App.TFrame")
        metrics.grid(row=1, column=0, sticky="ew", pady=(0, 12))
        metrics.columnconfigure((0, 1), weight=1, uniform="metrics")
        self.metric_values = {}
        self._metric_card(metrics, 0, 0, "EYE ASPECT RATIO", "—", "EAR", "ear")
        self._metric_card(metrics, 0, 1, "BLINK RATE", "—", "per minute", "blink_rate")
        self._metric_card(metrics, 1, 0, "PERCLOS", "—", "eye closure · 60 sec", "perclos")
        self._metric_card(metrics, 1, 1, "HEAD POSE", "—", "direction", "head_pose")
        for card in metrics.winfo_children():
            card.grid_configure(padx=4, pady=4, sticky="nsew")

        settings = ttk.Frame(sidebar, style="Card.TFrame", padding=16)
        settings.grid(row=2, column=0, sticky="nsew")
        settings.columnconfigure(1, weight=1)
        settings.columnconfigure(4, weight=1)
        ttk.Label(settings, text="SESSION SETUP", style="CardTitle.TLabel").grid(
            row=0, column=0, columnspan=3, sticky="w", pady=(0, 10))

        self.driver_var = tk.StringVar(value=self.args.driver)
        self.camera_var = tk.StringVar(value=str(
            self.args.camera if self.args.camera is not None else self.cfg["camera"]["index"]))
        self.lighting_var = tk.StringVar(value=self.args.lighting or "")
        self.position_var = tk.StringVar(value=self.args.camera_position or "")
        self.glasses_var = tk.StringVar(value=self.args.glasses or "")
        self.video_var = tk.StringVar(value=self.args.video or "")
        self.true_label_var = tk.StringVar(value="Normal")

        self._field(settings, 1, "Driver ID", self.driver_var, label_column=0)
        self._field(settings, 1, "Camera", self.camera_var, label_column=3)
        self._field(settings, 2, "Lighting", self.lighting_var,
                    values=("", "bright", "dim", "night"), label_column=0)
        self._field(settings, 2, "Position", self.position_var,
                    values=("", "front", "left-45", "right-45"), label_column=3)
        self._field(settings, 3, "Glasses", self.glasses_var,
                    values=("", "yes", "no"), label_column=0)

        ttk.Label(settings, text="Video", style="Muted.TLabel").grid(
            row=3, column=3, sticky="w", pady=4)
        video_entry = ttk.Entry(settings, textvariable=self.video_var)
        video_entry.grid(row=3, column=4, sticky="ew", padx=(8, 6), pady=4)
        browse = ttk.Button(settings, text="Browse", style="Secondary.TButton",
                            command=self._browse_video)
        browse.grid(row=3, column=5, sticky="e", pady=4)
        self.settings_widgets.extend((video_entry, browse))

        options = ttk.Frame(settings, style="Card.TFrame")
        options.grid(row=4, column=0, columnspan=6, sticky="ew", pady=(8, 2))
        for column in range(3):
            options.columnconfigure(column, weight=1)
        self.test_check = ttk.Checkbutton(options, text="Test mode (not saved)", variable=self.test_mode,
                                          command=self._update_test_label_state)
        self.test_check.grid(row=0, column=0, sticky="w")
        self.object_check = ttk.Checkbutton(options, text="Object detection (YOLO)",
                                            variable=self.object_detection)
        self.object_check.grid(row=0, column=1, sticky="w")
        self.audio_check = ttk.Checkbutton(options, text="Audio alerts",
                                           variable=self.alerts_enabled)
        self.audio_check.grid(row=0, column=2, sticky="w")
        self.settings_widgets.extend((self.test_check, self.object_check, self.audio_check))

        label_row = ttk.Frame(settings, style="Card.TFrame")
        label_row.grid(row=5, column=0, columnspan=6, sticky="ew", pady=(7, 0))
        ttk.Label(label_row, text="True label", style="Muted.TLabel").grid(row=0, column=0, sticky="w")
        self.true_label_combo = ttk.Combobox(
            label_row, textvariable=self.true_label_var, state="readonly",
            values=("Normal", "Drowsy", "Distracted", "Using Phone", "Eating", "Drinking"),
            width=18,
        )
        self.true_label_combo.grid(row=0, column=1, sticky="e", padx=(8, 0))
        self.true_label_combo.bind("<<ComboboxSelected>>", self._true_label_changed)
        self.settings_widgets.append(self.true_label_combo)
        self._update_test_label_state()

    def _metric_card(self, parent, row, column, title, initial, note, key):
        card = ttk.Frame(parent, style="Metric.TFrame", padding=(12, 10))
        card.grid(row=row, column=column, sticky="nsew")
        ttk.Label(card, text=title, style="MetricTitle.TLabel").grid(row=0, column=0, sticky="w")
        value = ttk.Label(card, text=initial, style="MetricValue.TLabel")
        value.grid(row=1, column=0, sticky="w", pady=(5, 0))
        ttk.Label(card, text=note, style="MetricNote.TLabel").grid(row=2, column=0, sticky="w")
        self.metric_values[key] = value

    def _field(self, parent, row, label, variable, values=None, label_column=0):
        field_column = label_column + 1
        ttk.Label(parent, text=label, style="Muted.TLabel").grid(
            row=row, column=label_column, sticky="w", pady=4)
        if values is None:
            widget = ttk.Entry(parent, textvariable=variable)
        else:
            widget = ttk.Combobox(parent, textvariable=variable, values=values,
                                  state="readonly")
        widget.grid(row=row, column=field_column, sticky="ew", padx=(8, 10), pady=4)
        self.settings_widgets.append(widget)

    def _browse_video(self):
        path = filedialog.askopenfilename(
            title="Choose a video",
            filetypes=(("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")),
        )
        if path:
            self.video_var.set(path)

    def _update_test_label_state(self):
        state = "readonly" if self.test_mode.get() and self.worker is None else "disabled"
        self.true_label_combo.configure(state=state)

    def _true_label_changed(self, _event=None):
        labels = ("Normal", "Drowsy", "Distracted", "Using Phone", "Eating", "Drinking")
        if self.worker is not None and self.test_mode.get():
            index = labels.index(self.true_label_var.get())
            self.worker.command(ord("0") + index)

    def _set_true_label(self, key):
        if self.worker is None or not self.test_mode.get():
            return
        labels = ("Normal", "Drowsy", "Distracted", "Using Phone", "Eating", "Drinking")
        index = int(key)
        self.true_label_var.set(labels[index])
        self.worker.command(ord("0") + index)

    def _hotkey(self, event, key):
        if isinstance(event.widget, (ttk.Entry, ttk.Combobox, tk.Entry, tk.Text)):
            return
        if key == "q":
            self._stop()
        elif key == "c":
            self._command(ord("c"))
        elif key == "m":
            self._toggle_audio()
        else:
            self._set_true_label(key)

    def _audio_button_text(self):
        return "Mute alerts" if self.alerts_enabled.get() else "Unmute alerts"

    def _toggle_audio(self):
        if self.worker is None:
            return
        self.alerts_enabled.set(not self.alerts_enabled.get())
        self.audio_button.configure(text=self._audio_button_text())
        self.worker.command(ord("m"))

    def _command(self, key):
        if self.worker is not None:
            self.worker.command(key)

    def _read_settings(self):
        driver = self.driver_var.get().strip()
        if not driver:
            raise ValueError("Enter a driver ID before starting.")
        try:
            camera = int(self.camera_var.get())
        except ValueError as exc:
            raise ValueError("Camera must be a non-negative number.") from exc
        if camera < 0:
            raise ValueError("Camera must be a non-negative number.")
        return {
            "driver": driver,
            "camera": camera,
            "video": self.video_var.get().strip(),
            "lighting": self.lighting_var.get(),
            "camera_position": self.position_var.get(),
            "glasses": self.glasses_var.get(),
            "test_mode": self.test_mode.get(),
            "objects": self.object_detection.get(),
            "audio": self.alerts_enabled.get(),
        }

    def _start(self):
        try:
            settings = self._read_settings()
            self.worker = self.worker_factory(settings)
            self.worker.start()
        except Exception as exc:
            self.worker = None
            messagebox.showerror("Could not start monitoring", str(exc), parent=self.root)
            return
        self.last_frame = None
        self.session_label.configure(text="Preparing session...")
        self.state_label.configure(text="Starting", foreground=MUTED)
        self.confidence_label.configure(text="Loading models and opening source")
        self._set_controls(running=True)
        self._set_connection("STARTING", MUTED)

    def _stop(self):
        if self.worker is not None:
            self.worker.stop()
            self.stop_button.configure(state="disabled")
            message = (
                "Stopping test run without saving..."
                if self.test_mode.get()
                else "Stopping safely and saving session..."
            )
            self.message_label.configure(text=message)

    def _set_controls(self, running):
        for widget in self.settings_widgets:
            widget.configure(state="disabled" if running else "normal")
        if not running:
            for widget in self.settings_widgets:
                if isinstance(widget, ttk.Combobox):
                    widget.configure(state="readonly")
            self._update_test_label_state()
        self.start_button.configure(state="disabled" if running else "normal")
        self.stop_button.configure(state="normal" if running else "disabled")
        self.recalibrate_button.configure(state="normal" if running else "disabled")
        self.audio_button.configure(state="normal" if running else "disabled")

    def _poll(self):
        if self.worker is not None:
            snapshot = self.worker.snapshot()
            if snapshot["frame"] is not None and snapshot["frame"] is not self.last_frame:
                self.last_frame = snapshot["frame"]
                self._show_frame(snapshot["frame"])
            if snapshot["metrics"]:
                self._show_metrics(snapshot["metrics"])
            self.message_label.configure(text=snapshot["message"])
            if snapshot["session_id"]:
                self.session_label.configure(text=f"SESSION  {snapshot['session_id']}")
            if snapshot["finished"]:
                error = snapshot["error"]
                self.worker = None
                self._set_controls(running=False)
                self._set_connection("ERROR" if error else "READY", RED if error else ACCENT)
                if error:
                    self.state_label.configure(text="Error", foreground=RED)
                    self.confidence_label.configure(text="Monitoring could not continue")
                    if not self.closing:
                        messagebox.showerror("Monitoring error", error, parent=self.root)
                elif not self.closing:
                    label = "Session saved" if snapshot["saved"] else "Session not saved"
                    self.confidence_label.configure(text=label)
                if self.closing:
                    self.root.destroy()
                    return
        if self.closing:
            self.root.destroy()
            return
        self.root.after(40, self._poll)

    def _show_frame(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        width = max(1, self.preview.winfo_width())
        height = max(1, self.preview.winfo_height())
        image.thumbnail((width, height), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(image)
        self.preview.configure(image=photo, text="")
        self.preview.image = photo

    def _show_metrics(self, metrics):
        behavior = metrics.get("behavior", "Ready")
        risky = behavior not in ("Ready", "Starting", "Calibrating", "Normal")
        color = RED if risky else ACCENT
        self.state_label.configure(text=behavior, foreground=color)
        confidence = metrics.get("confidence", 0.0)
        if behavior == "Calibrating":
            progress = metrics.get("calibration_progress", 0.0)
            detail = f"Calibration · {progress:.0%}"
        else:
            detail = f"Confidence · {confidence:.0%}"
        self.confidence_label.configure(text=detail)

        ear = metrics.get("ear")
        self.metric_values["ear"].configure(text=f"{ear:.3f}" if ear is not None else "—")
        self.metric_values["blink_rate"].configure(
            text=f"{metrics.get('blink_rate', 0.0):.0f}")
        self.metric_values["perclos"].configure(
            text=f"{metrics.get('perclos', 0.0):.0%}")
        self.metric_values["head_pose"].configure(text=metrics.get("head_pose", "—"))
        self.message_label.configure(
            text=f"{metrics.get('fps', 0.0):.0f} FPS  ·  "
                 f"{'Face detected' if metrics.get('face_detected') else 'Face not detected'}")
        if behavior == "Calibrating":
            self._set_connection("CALIBRATING", "#edbc55")
        else:
            self._set_connection("MONITORING", color)

    def _set_connection(self, text, color):
        self.connection_label.configure(text=f"●  {text}", fg=color)

    def _close(self):
        self.closing = True
        if self.worker is not None:
            self.worker.stop()
            message = (
                "Closing after test run..."
                if self.test_mode.get()
                else "Closing after the session is saved..."
            )
            self.message_label.configure(text=message)
            self.stop_button.configure(state="disabled")
        else:
            self.root.destroy()
