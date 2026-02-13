import tkinter as tk
import customtkinter as ctk
from tkinter import messagebox
import cv2
import threading
import sys
import os
import time
from collections import deque
import config
from detector import DetectionEngine

class DetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title(config.GUI_TITLE)
        self.root.geometry(config.GUI_SIZE)
        
        self.active_objects = set()
        self.patch_active = False
        self.running = False
        self.is_recording = False
        self.video_writer = None
        self.engine = None
        self.cap = None
        
        # Timing state
        self.last_time = time.time()
        self.frame_times = deque(maxlen=30) 
        self.current_fps = 0.0
        self.record_accumulated_time = 0.0
        self.record_time_per_frame = 1.0 / config.RECORDING_FPS
        
        self.setup_ui()
        self.root.bind('<space>', self.toggle_patch)
        self.log("Waiting for GUI...")
        self.root.after(500, self.start_initialization_thread)

    def setup_ui(self):
        ctk.CTkLabel(self.root, text=config.GUI_TEXT, font=config.GUI_TEXT_FONT).pack(pady=10)
        self.status_lbl = ctk.CTkLabel(self.root, text="Initializing...", text_color=config.GUI_STATUS_LBL_COLOR)
        self.status_lbl.pack(pady=5)
        
        frame_input = ctk.CTkFrame(self.root, fg_color="transparent")
        frame_input.pack(pady=5)
        ctk.CTkLabel(frame_input, text="Add Object:").pack(side=tk.LEFT)
        self.entry_obj = ctk.CTkEntry(frame_input, width=140)
        self.entry_obj.pack(side=tk.LEFT, padx=5)
        self.entry_obj.bind('<Return>', self.add_object)
        ctk.CTkButton(frame_input, text="Add", width=60, command=self.add_object).pack(side=tk.LEFT)
        
        frame_suggestions = ctk.CTkFrame(self.root, fg_color="transparent")
        frame_suggestions.pack(pady=2)
        ctk.CTkLabel(frame_suggestions, text="Quick Add:", font=("Arial", 10, "bold"), text_color="gray").pack(side=tk.LEFT, padx=5)
        for name in ["Mobile", "Cup", "Bottle", "Laptop"]:
            ctk.CTkButton(frame_suggestions, text=name, width=50, font=("Arial", 10), command=lambda n=name: self.add_object(name=n)).pack(side=tk.LEFT, padx=2)

        # Standard Listbox styled for Dark Mode
        self.listbox = tk.Listbox(self.root, height=8, bg="#2b2b2b", fg="white", borderwidth=0, highlightthickness=0)
        self.listbox.pack(pady=10, fill=tk.X, padx=20)
        
        ctk.CTkButton(self.root, text="Remove Selected", font=("Arial", 12), command=self.remove_object).pack(pady=2)
        ctk.CTkButton(self.root, text="Clear All", font=("Arial", 12), fg_color="#555555", hover_color="#333333", command=self.clear_all_objects).pack(pady=2)

        ctk.CTkLabel(self.root, text="Press [SPACE] to toggle Glasses", font=("Arial", 12, "bold")).pack(pady=5)

        record_frame = ctk.CTkFrame(self.root, fg_color="transparent")
        record_frame.pack(pady=10)
        self.btn_record = ctk.CTkButton(record_frame, text="Start Recording", fg_color="gray", font=("Arial", 12), command=self.toggle_recording)
        self.btn_record.pack()

        self.btn_stop = ctk.CTkButton(self.root, text="EXIT DEMO", fg_color="#8B0000", hover_color="#600000", font=("Arial", 12, "bold"), command=self.stop_demo)
        self.btn_stop.pack(pady=20, ipadx=20, ipady=5)

    def log(self, msg):
        print(f"[GUI] {msg}")
        self.root.after(0, lambda: self.status_lbl.configure(text=msg))

    def toggle_patch(self, event=None):
        self.patch_active = not self.patch_active
        self.log(f"Digital Glasses: {'ON' if self.patch_active else 'OFF'}")

    def toggle_recording(self):
        if not self.is_recording: self.start_recording()
        else: self.stop_recording()

    def start_recording(self):
        if not os.path.exists(config.RECORDING_FOLDER):
            os.makedirs(config.RECORDING_FOLDER)
            
        filename = f"{config.RECORDING_FOLDER}/demo_{int(time.time())}{config.RECORDING_EXT}"
        frame_size = (config.FRAME_WIDTH, config.FRAME_HEIGHT)
        
        self.record_accumulated_time = 0.0

        try:
            fourcc = cv2.VideoWriter_fourcc(*config.CODEC_PRIMARY)
            self.video_writer = cv2.VideoWriter(filename, fourcc, config.RECORDING_FPS, frame_size)
            if not self.video_writer.isOpened(): raise Exception("Primary Codec Failed")
            self.log(f"Recording ({config.CODEC_PRIMARY})...")
        except Exception:
            fourcc = cv2.VideoWriter_fourcc(*config.CODEC_FALLBACK)
            self.video_writer = cv2.VideoWriter(filename, fourcc, config.RECORDING_FPS, frame_size)
            self.log(f"Recording ({config.CODEC_FALLBACK})...")

        self.is_recording = True
        self.btn_record.configure(text="Stop Recording", fg_color="red")

    def stop_recording(self):
        self.is_recording = False
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None
        self.btn_record.configure(text="Start Recording", fg_color="gray")
        self.log("Recording Saved.")

    def add_object(self, event=None, name=None):
        if name:
            raw_obj = name.strip().lower()
        else:
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

    def remove_object(self):
        selection = self.listbox.curselection()
        if not selection: return
        
        index = selection[0]
        text = self.listbox.get(index)
        
        # Extract resolved name from "raw (resolved)" or just "resolved"
        resolved_name = text.split('(')[-1].strip(')') if '(' in text else text
        
        if resolved_name in self.active_objects:
            self.active_objects.remove(resolved_name)
        
        self.listbox.delete(index)
        self.log(f"Removed '{resolved_name}'")

    def clear_all_objects(self):
        self.active_objects.clear()
        self.listbox.delete(0, tk.END)
        self.log("Cleared all objects")

    def start_initialization_thread(self):
        t = threading.Thread(target=self._init_resources, daemon=True)
        t.start()

    def _init_resources(self):
        try:
            self.log("Loading AI Models...")
            self.engine = DetectionEngine() 
            self.log("Opening Camera...")
            if sys.platform == 'win32': self.cap = cv2.VideoCapture(config.CAMERA_ID, cv2.CAP_DSHOW)
            else: self.cap = cv2.VideoCapture(config.CAMERA_ID)
            
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
            
            if not self.cap.isOpened(): self.cap = cv2.VideoCapture(config.CAMERA_ID)
            if not self.cap.isOpened(): raise Exception("Cannot open webcam.")
            self.root.after(0, self._finalize_setup)
        except Exception as e:
            self.log(f"Error: {e}")
            self.root.after(0, lambda: messagebox.showerror("Init Error", str(e)))

    def _finalize_setup(self):
        cv2.namedWindow(config.WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.moveWindow(config.WINDOW_NAME, config.WINDOW_POS_X, config.WINDOW_POS_Y)
        self.running = True
        self.log("System Ready")
        self.last_time = time.time()
        self.update_loop()

    def update_loop(self):
        if not self.running: return
        
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
            
            if self.is_recording and self.video_writer:
                self.record_accumulated_time += delta
                while self.record_accumulated_time >= self.record_time_per_frame:
                    final_frame_record = final_frame.copy()
                    cv2.circle(final_frame_record, (30, 30), 10, config.COLOR_RECORD_DOT, -1)
                    self.video_writer.write(final_frame_record)
                    self.record_accumulated_time -= self.record_time_per_frame
            
            try: cv2.imshow(config.WINDOW_NAME, final_frame)
            except cv2.error: self.stop_demo(); return

        self.root.after(config.GUI_UPDATE_DELAY, self.update_loop)

    def stop_demo(self):
        self.running = False
        if self.is_recording: self.stop_recording()
        if self.cap and self.cap.isOpened(): self.cap.release()
        cv2.destroyAllWindows()
        self.root.quit()
        self.root.destroy()