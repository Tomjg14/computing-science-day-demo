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
        
        # --- FPS Calculation State ---
        self.last_time = time.time()
        # Store last 30 frame times to get a smooth average
        self.frame_times = deque(maxlen=30) 
        self.current_fps = 15.0 # Default fallback
        
        # --- GUI Layout ---
        self.setup_ui()
        
        # --- Bind Spacebar Globally ---
        self.root.bind('<space>', self.toggle_patch)
        
        # --- Initialize in Background ---
        self.log("Waiting for GUI...")
        self.root.after(500, self.start_initialization_thread)

    def setup_ui(self):
        # Header
        tk.Label(self.root, text="Adversarial Demo", font=("Arial", 16, "bold")).pack(pady=10)
        
        # Status
        self.status_lbl = tk.Label(self.root, text="Initializing...", fg="blue")
        self.status_lbl.pack(pady=5)
        
        # Object Input
        frame_input = tk.Frame(self.root)
        frame_input.pack(pady=5)
        tk.Label(frame_input, text="Add Object:").pack(side=tk.LEFT)
        self.entry_obj = tk.Entry(frame_input)
        self.entry_obj.pack(side=tk.LEFT, padx=5)
        self.entry_obj.bind('<Return>', self.add_object)
        tk.Button(frame_input, text="Add", command=self.add_object).pack(side=tk.LEFT)
        
        # Valid Objects Hint
        tk.Label(self.root, text="(Try: 'cell phone', 'bottle', 'cup', 'laptop')", 
                 font=("Arial", 8), fg="gray").pack(pady=0)
        
        # Listbox
        self.listbox = tk.Listbox(self.root, height=8)
        self.listbox.pack(pady=10, fill=tk.X, padx=20)
        
        # Instructions
        tk.Label(self.root, text="Press [SPACE] to toggle Glasses", fg="black", font=("Arial", 10, "bold")).pack(pady=5)

        # Recording Controls
        record_frame = tk.Frame(self.root)
        record_frame.pack(pady=10)
        
        self.btn_record = tk.Button(record_frame, text="Start Recording", bg="gray", fg="white", 
                                    font=("Arial", 10), command=self.toggle_recording)
        self.btn_record.pack()

        # Stop Button
        self.btn_stop = tk.Button(self.root, text="EXIT DEMO", bg="#8B0000", fg="white", 
                                  font=("Arial", 12, "bold"), command=self.stop_demo)
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
        
        # SMART FPS: Use the actual calculated FPS from the last few seconds
        # We ensure it's at least 5 to avoid errors if startup was slow
        record_fps = max(5.0, round(self.current_fps))
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(filename, fourcc, record_fps, (1280, 720))
        
        self.is_recording = True
        self.btn_record.config(text=f"Stop Rec ({int(record_fps)} FPS)", bg="red")
        self.log(f"Recording at {int(record_fps)} FPS...")

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
                messagebox.showwarning("Unknown Object", 
                    f"Could not understand '{raw_obj}'.\nTry standard names like 'bottle', 'cup', 'mouse'.")
                return
            
            if match_type == "alias":
                self.log(f"Mapped '{raw_obj}' -> '{resolved_name}'")
            elif match_type == "fuzzy":
                if not messagebox.askyesno("Did you mean?", f"Did you mean '{resolved_name}'?"):
                    return
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
            if sys.platform == 'win32':
                self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            else:
                self.cap = cv2.VideoCapture(0)
            
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            
            if not self.cap.isOpened():
                self.cap = cv2.VideoCapture(0)
            
            if not self.cap.isOpened():
                raise Exception("Cannot open webcam.")

            self.root.after(0, self._finalize_setup)
            
        except Exception as e:
            err_msg = str(e)
            self.log(f"Error: {err_msg}")
            self.root.after(0, lambda: messagebox.showerror("Init Error", err_msg))

    def _finalize_setup(self):
        cv2.namedWindow(self.win_name, cv2.WINDOW_NORMAL)
        cv2.moveWindow(self.win_name, 1920, 0)
        self.running = True
        self.log("System Ready")
        
        # Reset timer before loop starts
        self.last_time = time.time()
        self.update_loop()

    def update_loop(self):
        if not self.running:
            return
        
        # --- FPS CALCULATION ---
        current_time = time.time()
        delta = current_time - self.last_time
        self.last_time = current_time
        
        # Avoid division by zero
        if delta > 0:
            self.frame_times.append(delta)
            # Calculate average FPS over last 30 frames
            avg_delta = sum(self.frame_times) / len(self.frame_times)
            if avg_delta > 0:
                self.current_fps = 1.0 / avg_delta

        # --- KEY INPUT ---
        key = cv2.waitKey(1) & 0xFF
        if key == 27: 
            self.stop_demo()
            return

        # --- PROCESSING ---
        ret, frame = self.cap.read()
        if ret:
            final_frame = self.engine.process_frame(frame, self.active_objects, self.patch_active)
            
            if self.is_recording and self.video_writer:
                self.video_writer.write(final_frame)
                cv2.circle(final_frame, (30, 30), 10, (0, 0, 255), -1)

            try:
                cv2.imshow(self.win_name, final_frame)
            except cv2.error:
                self.stop_demo()
                return

        self.root.after(10, self.update_loop)

    def stop_demo(self):
        self.running = False
        self.stop_recording() 
        if self.cap and self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()
        self.root.quit()
        self.root.destroy()