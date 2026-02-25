import tkinter as tk
import customtkinter as ctk
from tkinter import messagebox
import cv2
import threading
import sys
import os
import time
import numpy as np
from collections import deque
import config
from detector import DetectionEngine
from PIL import Image, ImageTk

class DetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title(config.GUI_TITLE)

        # --- SCALING ---
        self.scale_factor = config.GUI_SCALE_FACTOR
        width, height = map(int, config.GUI_SIZE.split('x'))
        scaled_size = f"{self.scale(width)}x{self.scale(height)}"
        self.root.geometry(scaled_size)
        # --- END SCALING ---
        
        self.active_objects = set()
        self.patch_active = False
        self.running = False
        self.is_recording = False
        self.video_writer = None
        self.engine = None
        self.cap = None
        self.current_frame = None
        self.current_camera_index = config.CAMERA_ID
        self.video_thread = None
        
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

    def scale(self, value):
        """Scales a numeric value or a tuple/list of values."""
        if isinstance(value, (tuple, list)):
            return [int(v * self.scale_factor) for v in value]
        return int(value * self.scale_factor)

    def scale_font(self, font_tuple):
        """Scales the size part of a font tuple."""
        family, size, *style = font_tuple
        # Re-join style elements like "bold", "italic"
        return (family, self.scale(size), *style)

    def setup_ui(self):
        self._setup_main_layout()
        self._create_header()
        self._create_input_section()
        self._create_object_controls()
        self._create_face_recognition_section()
        self._create_footer_controls()

    def _setup_main_layout(self):
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_columnconfigure(1, weight=0)
        self.root.grid_columnconfigure(2, weight=1)
        self.root.grid_rowconfigure(0, weight=1)
        
        self.ui_box = ctk.CTkFrame(self.root, fg_color="transparent")
        self.ui_box.grid(row=0, column=1, sticky="nsew", pady=self.scale(10))
        self.BTN_WIDTH = self.scale(320)

    def _create_header(self):
        scaled_font = self.scale_font(config.GUI_TEXT_FONT)
        ctk.CTkLabel(self.ui_box, text=config.GUI_TEXT, font=scaled_font).pack(pady=self.scale(10))
        self.status_lbl = ctk.CTkLabel(self.ui_box, text="Initializing...", text_color=config.GUI_STATUS_LBL_COLOR)
        self.status_lbl.pack(pady=self.scale(5))

    def _create_input_section(self):
        frame_input = ctk.CTkFrame(self.ui_box, fg_color="transparent")
        frame_input.pack(pady=self.scale(5), fill="x")
        ctk.CTkLabel(frame_input, text="Add Object:").pack(side=tk.LEFT)
        self.entry_obj = ctk.CTkEntry(frame_input, width=self.scale(140))
        self.entry_obj.pack(side=tk.LEFT, padx=self.scale(5), fill="x", expand=True)
        self.entry_obj.bind('<Return>', self.add_object)
        ctk.CTkButton(frame_input, text="Add", width=self.scale(60), command=self.add_object).pack(side=tk.LEFT)
        
        frame_suggestions = ctk.CTkFrame(self.ui_box, fg_color="transparent")
        frame_suggestions.pack(pady=self.scale(2), fill="x")
        ctk.CTkLabel(frame_suggestions, text="Quick Add:", font=("Arial", self.scale(10), "bold"), text_color="gray").pack(side=tk.LEFT, padx=(0, self.scale(5)))
        for name in ["Mobile", "Cup", "Bottle", "Laptop"]:
            ctk.CTkButton(frame_suggestions, text=name, width=self.scale(50), font=("Arial", self.scale(10)), command=lambda n=name: self.add_object(name=n)).pack(side=tk.LEFT, padx=self.scale(2))

    def _create_object_controls(self):
        ctk.CTkButton(self.ui_box, text="Manage Objects", width=self.BTN_WIDTH, command=self.open_object_manager).pack(pady=self.scale(10))

    def _create_face_recognition_section(self):
        ctk.CTkLabel(self.ui_box, text="Face Recognition", font=("Arial", self.scale(12), "bold")).pack(pady=self.scale((15, 5)))
        self.lbl_face_count = ctk.CTkLabel(self.ui_box, text="Known Faces: 0", text_color="gray", font=("Arial", self.scale(11)))
        self.lbl_face_count.pack(pady=self.scale((0, 5)))
        self.btn_capture_face = ctk.CTkButton(self.ui_box, text="Capture & Name Face", width=self.BTN_WIDTH, fg_color="#2E8B57", hover_color="#206040", command=self.open_capture_window)
        self.btn_capture_face.pack(pady=self.scale(2))
        self.btn_manage_faces = ctk.CTkButton(self.ui_box, text="Manage Database", width=self.BTN_WIDTH, fg_color="#444444", hover_color="#333333", command=self.open_face_manager).pack(pady=self.scale(2))

    def _create_footer_controls(self):
        ctk.CTkLabel(self.ui_box, text="[SPACE] Glasses | [F] Fullscreen", font=("Arial", self.scale(12), "bold")).pack(pady=self.scale(5))
        record_frame = ctk.CTkFrame(self.ui_box, fg_color="transparent")
        record_frame.pack(pady=self.scale(10), fill="x")
        self.btn_record = ctk.CTkButton(record_frame, text="Start Recording", fg_color="gray", font=("Arial", self.scale(12)), command=self.toggle_recording)
        self.btn_record.pack(fill="x")
        ctk.CTkButton(self.ui_box, text="Switch Camera", width=self.BTN_WIDTH, font=("Arial", self.scale(12)), fg_color="#444444", hover_color="#333333", command=self.open_camera_selector).pack(pady=self.scale(5))
        self.btn_stop = ctk.CTkButton(self.ui_box, text="EXIT DEMO", width=self.BTN_WIDTH, fg_color="#8B0000", hover_color="#600000", font=("Arial", 12, "bold"), command=self.stop_demo)
        self.btn_stop.pack(pady=self.scale(20), ipadx=self.scale(20), ipady=self.scale(5))

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
                self.log(f"Added '{resolved_name}'")
                self.entry_obj.delete(0, tk.END)

    def open_object_manager(self):
        top = ctk.CTkToplevel(self.root)
        top.title("Manage Objects")
        top.geometry(f"{self.scale(300)}x{self.scale(400)}")
        top.attributes("-topmost", True)

        lb = tk.Listbox(top, bg="#2b2b2b", fg="white", borderwidth=0, highlightthickness=0, font=self.scale_font(("Arial", 12)))
        lb.pack(fill=tk.BOTH, expand=True, padx=self.scale(10), pady=self.scale(10))
        
        for name in sorted(list(self.active_objects)):
            lb.insert(tk.END, name)

        def delete_selected():
            sel = lb.curselection()
            if not sel: return
            name = lb.get(sel[0])
            if name in self.active_objects:
                self.active_objects.remove(name)
                lb.delete(sel[0])
                self.log(f"Removed '{name}'")

        def clear_all():
            self.active_objects.clear()
            lb.delete(0, tk.END)
            self.log("Cleared all objects")

        ctk.CTkButton(top, text="Delete Selected", font=("Arial", self.scale(12)), fg_color="#8B0000", hover_color="#600000", command=delete_selected).pack(pady=self.scale(5))
        ctk.CTkButton(top, text="Clear All", font=("Arial", self.scale(12)), fg_color="#555555", hover_color="#333333", command=clear_all).pack(pady=(0, self.scale(10)))

    def update_face_count(self):
        if self.engine:
            count = len(self.engine.known_face_names)
            self.lbl_face_count.configure(text=f"Known Faces: {count}")

    def open_face_manager(self):
        if not self.engine: return
        
        top = ctk.CTkToplevel(self.root)
        top.title("Manage Faces")
        top.geometry(f"{self.scale(300)}x{self.scale(400)}")
        top.attributes("-topmost", True)

        lb = tk.Listbox(top, bg="#2b2b2b", fg="white", borderwidth=0, highlightthickness=0, font=self.scale_font(("Arial", 12)))
        lb.pack(fill=tk.BOTH, expand=True, padx=self.scale(10), pady=self.scale(10))
        
        for name in self.engine.known_face_names:
            lb.insert(tk.END, name)

        def delete_selected():
            sel = lb.curselection()
            if not sel: return
            name = lb.get(sel[0])
            if self.engine.remove_face(name):
                lb.delete(sel[0])
                self.update_face_count()
                messagebox.showinfo("Removed", f"Removed '{name}' from database.", parent=top)

        def clear_all():
            if messagebox.askyesno("Confirm", "Are you sure you want to delete ALL known faces?", parent=top):
                self.engine.clear_known_faces()
                lb.delete(0, tk.END)
                self.update_face_count()
                messagebox.showinfo("Cleared", "Database cleared.", parent=top)

        ctk.CTkButton(top, text="Delete Selected", font=("Arial", self.scale(12)), fg_color="#8B0000", hover_color="#600000", command=delete_selected).pack(pady=self.scale(5))
        ctk.CTkButton(top, text="Clear Database", font=("Arial", self.scale(12)), fg_color="#555555", hover_color="#333333", command=clear_all).pack(pady=(0, self.scale(10)))

    def open_capture_window(self):
        if self.current_frame is None or not self.engine:
            messagebox.showwarning("Error", "No video feed available.")
            return

        # Find the largest face in the current frame
        results = self.engine.model_face(self.current_frame, verbose=False, conf=config.CONF_THRESHOLD_FACE)
        best_box = None
        max_area = 0
        
        for r in results:
            for box in r.boxes:
                if int(box.cls[0]) == 0:
                    x, y, w, h = self.engine.get_coords(box)
                    if w * h > max_area:
                        max_area = w * h
                        best_box = (x, y, w, h)
        
        if not best_box:
            messagebox.showinfo("No Face", "No face detected in the current frame.")
            return

        # Create Popup
        top = ctk.CTkToplevel(self.root)
        top.title("Save Face")
        top.geometry(f"{self.scale(300)}x{self.scale(350)}")
        top.attributes("-topmost", True)

        x, y, w, h = best_box
        face_crop = self.current_frame[y:y+h, x:x+w]
        face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(face_rgb)
        
        pil_img.thumbnail((self.scale(200), self.scale(200)))
        tk_img = ImageTk.PhotoImage(pil_img)

        ctk.CTkLabel(top, text="Detected Face:", font=("Arial", self.scale(12))).pack(pady=self.scale(10))
        lbl_img = ctk.CTkLabel(top, text="", image=tk_img)
        lbl_img.image = tk_img # Keep ref
        lbl_img.pack(pady=self.scale(5))

        entry_name = ctk.CTkEntry(top, placeholder_text="Enter Name", width=self.scale(160), font=("Arial", self.scale(12)))
        entry_name.pack(pady=self.scale(10))
        entry_name.focus()

        def save_action():
            name = entry_name.get().strip()
            if not name: return
            success, msg = self.engine.register_face(self.current_frame, best_box, name)
            if success:
                self.log(msg)
                self.update_face_count()
                top.destroy()
            else:
                messagebox.showerror("Error", msg)

        ctk.CTkButton(top, text="Save", font=("Arial", self.scale(12)), command=save_action).pack(pady=self.scale(10))

    def get_available_cameras(self):
        """Scans for available cameras (indices 0-3)."""
        available = []
        # Check first 4 indices
        for i in range(4):
            if i == self.current_camera_index:
                available.append(i)
                continue

            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW if sys.platform == 'win32' else None)
            if cap.isOpened():
                available.append(i)
                cap.release()
        return available

    def open_camera_selector(self):
        top = ctk.CTkToplevel(self.root)
        top.title("Select Camera")
        top.geometry(f"{self.scale(250)}x{self.scale(300)}")
        top.attributes("-topmost", True)

        ctk.CTkLabel(top, text="Available Cameras:", font=("Arial", self.scale(12), "bold")).pack(pady=self.scale(10))
        
        cams = self.get_available_cameras()
        if not cams:
            ctk.CTkLabel(top, text="No cameras found.", font=("Arial", self.scale(12))).pack(pady=10)
        
        for idx in cams:
            text = f"Camera {idx}"
            if idx == self.current_camera_index: text += " (Active)"
            
            btn = ctk.CTkButton(top, text=text, font=("Arial", self.scale(12)),
                                command=lambda i=idx: [self.change_camera(i), top.destroy()])
            btn.pack(pady=self.scale(5))

    def change_camera(self, index):
        # Ignore if switching to the same camera that is currently running
        if int(index) == int(self.current_camera_index) and self.running and self.video_thread and self.video_thread.is_alive():
            self.log(f"Camera {index} is already active.")
            return
        
        self.log(f"Switching to Camera {index}...")
        self.running = False # Stop current loop
        if self.video_thread and self.video_thread.is_alive():
            self.video_thread.join() # Wait for loop to exit
        
        # Give the OS a moment to release the camera resource
        time.sleep(0.5)
            
        self.current_camera_index = index
        self.video_thread = threading.Thread(target=self._restart_video_stream, daemon=True)
        self.video_thread.start()

    def _restart_video_stream(self):
        self._open_camera()
        self._video_loop()

    def start_initialization_thread(self):
        self.video_thread = threading.Thread(target=self._init_resources, daemon=True)
        self.video_thread.start()

    def _init_resources(self):
        try:
            self.log("Loading AI Models...")
            self.engine = DetectionEngine() 
            
            self._open_camera()
            if not self.cap.isOpened(): 
                raise Exception("Cannot open webcam.")
            
            # Signal GUI that we are ready (must be done on main thread)
            self.root.after(0, self._finalize_gui_setup)
            
            # Start the video loop in this background thread
            self._video_loop()
            
        except Exception as e:
            self.log(f"Error: {e}")
            self.root.after(0, lambda: messagebox.showerror("Init Error", str(e)))

    def _open_camera(self):
        self.log(f"Opening Camera {self.current_camera_index}...")
        if self.cap: self.cap.release()
        
        if sys.platform == 'win32': 
            self.cap = cv2.VideoCapture(self.current_camera_index, cv2.CAP_DSHOW)
        else: 
            self.cap = cv2.VideoCapture(self.current_camera_index)
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

    def _finalize_gui_setup(self):
        """
        Finalizes GUI elements on the main thread after resources are loaded.
        """
        self.running = True
        
        # Update GUI state based on loaded resources
        if not self.engine.HAS_FACE_REC:
            self.btn_capture_face.configure(state=tk.DISABLED, text="Capture (face_recognition missing)")
            self.log("Face recognition disabled. See console.")
        self.update_face_count()
        self.log("System Ready")

    def _video_loop(self):
        """
        Runs the video processing in a separate thread to keep the GUI responsive.
        """
        # Setup OpenCV window in this thread
        cv2.namedWindow(config.WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.moveWindow(config.WINDOW_NAME, config.WINDOW_POS_X, config.WINDOW_POS_Y)
        if config.WINDOW_FULLSCREEN:
            cv2.setWindowProperty(config.WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        
        self.last_time = time.time()
        self.running = True

        while self.running:
            current_time = time.time()
            delta = current_time - self.last_time
            self.last_time = current_time

            key = cv2.waitKey(1) & 0xFF
            if key == 27: # ESC
                self.root.after(0, self.stop_demo)
                break
            elif key == ord('f'):
                prop = cv2.getWindowProperty(config.WINDOW_NAME, cv2.WND_PROP_FULLSCREEN)
                if prop == cv2.WINDOW_FULLSCREEN:
                    cv2.setWindowProperty(config.WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                else:
                    cv2.setWindowProperty(config.WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

            if self.cap and self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret:
                    self.current_frame = frame.copy() # Store for capture
                    final_frame = self.engine.process_frame(frame, self.active_objects, self.patch_active)
                    
                    if self.is_recording and self.video_writer:
                        self.record_accumulated_time += delta
                        while self.record_accumulated_time >= self.record_time_per_frame:
                            final_frame_record = final_frame.copy()
                            cv2.circle(final_frame_record, (30, 30), 10, config.COLOR_RECORD_DOT, -1)
                            self.video_writer.write(final_frame_record)
                            self.record_accumulated_time -= self.record_time_per_frame
                else:
                    # Handle camera not ready or no signal
                    final_frame = np.zeros((config.FRAME_HEIGHT, config.FRAME_WIDTH, 3), dtype=np.uint8)
                    cv2.putText(final_frame, "No Signal / Loading...", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    time.sleep(0.1) # Prevent CPU spin
                    
                try: cv2.imshow(config.WINDOW_NAME, final_frame)
                except cv2.error: 
                    self.root.after(0, self.stop_demo)
                    break
        
        # Cleanup when loop exits
        if self.cap: self.cap.release()
        cv2.destroyAllWindows()


    def stop_demo(self):
        self.running = False
        if self.engine:
            self.engine.clear_known_faces() # Clear database on exit
        if self.is_recording: self.stop_recording()
        # Note: cap release and destroyAllWindows are now handled in the thread
        self.root.quit()
        # self.root.destroy() # quit() is usually sufficient and safer with threads