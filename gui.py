import tkinter as tk
from tkinter import messagebox
import cv2
import threading
import sys
import os
import time
from collections import deque
from detector import DetectionEngine

class DetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Control Panel")
        self.root.geometry("400x550")
        
        # --- State ---
        self.active_objects = set()
        self.patch_active = False
        self.running = False
        self.is_recording = False
        self.video_writer = None
        self.engine = None
        self.cap = None
        self.win_name = "Live Detection Stream"
        
        # --- FPS & Timing State ---
        self.last_time = time.time()
        self.frame_times = deque(maxlen=30) 
        self.current_fps = 0.0
        
        # --- Recording Sync State ---
        self.record_target_fps = 20.0          # We force the FILE to be 20 FPS
        self.record_time_per_frame = 1.0 / 20.0 
        self.record_accumulated_time = 0.0     # Tracks how much "real time" has passed
        
        self.setup_ui()
        self.root.bind('<space>', self.toggle_patch)
        self.log("Waiting for GUI...")
        self.root.after(500, self.start_initialization_thread)

    def setup_ui(self):
        tk.Label(self.root, text="Adversarial Demo", font=("Arial", 16, "bold")).pack(pady=10)
        self.status_lbl = tk.Label(self.root, text="Initializing...", fg="blue")
        self.status_lbl.pack(pady=5)
        
        frame_input = tk.Frame(self.root)
        frame_input.pack(pady=5)
        tk.Label(frame_input, text="Add Object:").pack(side=tk.LEFT)
        self.entry_obj = tk.Entry(frame_input)
        self.entry_obj.pack(side=tk.LEFT, padx=5)
        self.entry_obj.bind('<Return>', self.add_object)
        tk.Button(frame_input, text="Add", command=self.add_object).pack(side=tk.LEFT)
        
        tk.Label(self.root, text="(Try: 'cell phone', 'bottle', 'cup', 'laptop')", font=("Arial", 8), fg="gray").pack(pady=0)
        self.listbox = tk.Listbox(self.root, height=8)
        self.listbox.pack(pady=10, fill=tk.X, padx=20)
        tk.Label(self.root, text="Press [SPACE] to toggle Glasses", fg="black", font=("Arial", 10, "bold")).pack(pady=5)

        record_frame = tk.Frame(self.root)
        record_frame.pack(pady=10)
        self.btn_record = tk.Button(record_frame, text="Start Recording", bg="gray", fg="white", font=("Arial", 10), command=self.toggle_recording)
        self.btn_record.pack()

        self.btn_stop = tk.Button(self.root, text="EXIT DEMO", bg="#8B0000", fg="white", font=("Arial", 12, "bold"), command=self.stop_demo)
        self.btn_stop.pack(pady=20, ipadx=20, ipady=5)

    def log(self, msg):
        print(f"[GUI] {msg}")
        self.root.after(0, lambda: self.status_lbl.config(text=msg))

    def toggle_patch(self, event=None):
        self.patch_active = not self.patch_active
        state = "ON" if self.patch_active else "OFF"
        self.log(f"Digital Glasses: {state}")

    def toggle_recording(self):
        if not self.is_recording:
            self.start_recording()
        else:
            self.stop_recording()

    def start_recording(self):
        if not os.path.exists("recordings"):
            os.makedirs("recordings")
            
        filename = f"recordings/demo_{int(time.time())}.mp4"
        frame_size = (1280, 720) 
        
        # Reset the accumulator so we start fresh
        self.record_accumulated_time = 0.0

        # Try H.264 (avc1) first, fallback to mp4v
        try:
            fourcc = cv2.VideoWriter_fourcc(*'avc1')
            self.video_writer = cv2.VideoWriter(filename, fourcc, self.record_target_fps, frame_size)
            if not self.video_writer.isOpened(): raise Exception("H.264 failed")
            self.log(f"Recording (H.264) started...")
        except Exception:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(filename, fourcc, self.record_target_fps, frame_size)
            self.log(f"Recording (MP4V) started...")

        self.is_recording = True
        self.btn_record.config(text="Stop Recording", bg="red")

    def stop_recording(self):
        self.is_recording = False
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None
        self.btn_record.config(text="Start Recording", bg="gray")
        self.log("Recording Saved.")

    def add_object(self, event=None):
        raw_obj = self.entry_obj.get().strip().lower()
        if not raw_obj: return

        if self.engine:
            resolved_name, match_type = self.engine.resolve_object_name(raw_obj)

            if match_type == "fail":
                messagebox.showwarning("Unknown Object", f"Could not understand '{raw_obj}'.")
                return
            
            if match_type == "alias": self.log(f"Mapped '{raw_obj}' -> '{resolved_name}'")
            elif match_type == "fuzzy":
                if not messagebox.askyesno("Did you mean?", f"Did you mean '{resolved_name}'?"): return
                self.log(f"Corrected '{raw_obj}' -> '{resolved_name}'")

            if resolved_name not in self.active_objects:
                self.active_objects.add(resolved_name)
                display_text = resolved_name if match_type != "alias" else f"{raw_obj} ({resolved_name})"
                self.listbox.insert(tk.END, display_text)
                self.entry_obj.delete(0, tk.END)

    def start_initialization_thread(self):
        t = threading.Thread(target=self._init_resources, daemon=True)
        t.start()

    def _init_resources(self):
        try:
            self.log("Loading AI Models...")
            self.engine = DetectionEngine() 
            self.log("Opening Camera...")
            if sys.platform == 'win32': self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            else: self.cap = cv2.VideoCapture(0)
            
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            
            if not self.cap.isOpened(): self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened(): raise Exception("Cannot open webcam.")
            self.root.after(0, self._finalize_setup)
        except Exception as e:
            self.log(f"Error: {e}")
            self.root.after(0, lambda: messagebox.showerror("Init Error", str(e)))

    def _finalize_setup(self):
        cv2.namedWindow(self.win_name, cv2.WINDOW_NORMAL)
        cv2.moveWindow(self.win_name, 1920, 0)
        self.running = True
        self.log("System Ready")
        self.last_time = time.time()
        self.update_loop()

    def update_loop(self):
        if not self.running: return
        
        # --- Time Delta Calculation ---
        current_time = time.time()
        delta = current_time - self.last_time
        self.last_time = current_time

        key = cv2.waitKey(1) & 0xFF
        if key == 27: 
            self.stop_demo()
            return

        ret, frame = self.cap.read()
        if ret:
            final_frame = self.engine.process_frame(frame, self.active_objects, self.patch_active)
            
            # --- SYNCHRONIZED RECORDING ---
            if self.is_recording and self.video_writer:
                # Add the time passed since last loop to our "bucket"
                self.record_accumulated_time += delta
                
                # While we have enough time in the bucket for a frame (0.05s)
                while self.record_accumulated_time >= self.record_time_per_frame:
                    # Write the frame
                    final_frame_record = final_frame.copy()
                    cv2.circle(final_frame_record, (30, 30), 10, (0, 0, 255), -1)
                    self.video_writer.write(final_frame_record)
                    
                    # Remove that chunk of time from the bucket
                    self.record_accumulated_time -= self.record_time_per_frame
            
            try: cv2.imshow(self.win_name, final_frame)
            except cv2.error: self.stop_demo(); return

        self.root.after(10, self.update_loop)

    def stop_demo(self):
        self.running = False
        if self.is_recording: self.stop_recording()
        if self.cap and self.cap.isOpened(): self.cap.release()
        cv2.destroyAllWindows()
        self.root.quit()
        self.root.destroy()