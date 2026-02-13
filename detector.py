import cv2
import torch
import os
import requests
import numpy as np
import math
import difflib
from ultralytics import YOLO
from patch_generator import PatchGenerator
from utils import resource_path  # Import the helper

class SmartResolver:
    """Helps find the correct COCO class name from user input."""
    def __init__(self, model_names):
        self.names = model_names 
        self.valid_list = [v.lower() for v in self.names.values()]
        
        self.aliases = {
            "mobile": "cell phone", "phone": "cell phone", 
            "iphone": "cell phone", "smartphone": "cell phone",
            "mug": "cup", "coffee": "cup", "tea": "cup",
            "coke": "bottle", "water": "bottle", "beer": "bottle", "can": "bottle",
            "screen": "tv", "monitor": "tv", "display": "tv",
            "pc": "laptop", "computer": "laptop", "macbook": "laptop",
            "controller": "remote", "gamepad": "remote",
            "man": "person", "woman": "person", "human": "person", 
            "guy": "person", "girl": "person", "boy": "person",
            "bike": "bicycle", "auto": "car", "taxi": "car"
        }

    def resolve(self, user_input):
        text = user_input.lower().strip()
        if text in self.valid_list: return text, "exact"
        if text in self.aliases: return self.aliases[text], "alias"
        matches = difflib.get_close_matches(text, self.valid_list, n=1, cutoff=0.6)
        if matches: return matches[0], "fuzzy"
        return None, "fail"

class DetectionEngine:
    def __init__(self, status_callback=None):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.status_callback = status_callback
        
        self.log(f"Loading Models on {self.device.upper()}...")
        
        # 1. Load Face Model (Check internal resources first)
        face_model_path = self.get_best_face_model()
        self.log(f"Loading Face Model: {face_model_path}")
        self.model_face = YOLO(face_model_path) 
        
        # 2. Load Object Model (Check internal resources first)
        obj_model_path = resource_path('yolov8n.pt')
        if not os.path.exists(obj_model_path):
            obj_model_path = 'yolov8n.pt' # Fallback to standard download if missing
            
        self.log(f"Loading Object Model: {obj_model_path}")
        self.model_objects = YOLO(obj_model_path) 
        
        self.resolver = SmartResolver(self.model_objects.names)
        self.glasses_img = PatchGenerator.get_glasses_mask()
        self.log("Models Loaded.")

    def log(self, msg):
        if self.status_callback:
            self.status_callback(msg)
        print(f"[ENGINE] {msg}")

    def get_best_face_model(self):
        filename = "yolov8n-face.pt"
        
        # 1. Check if bundled in the .exe (PyInstaller)
        bundled_path = resource_path(filename)
        if os.path.exists(bundled_path) and os.path.getsize(bundled_path) > 1000000:
            return bundled_path

        # 2. Check local folder (Dev mode)
        if os.path.exists(filename) and os.path.getsize(filename) > 1000000:
            return filename
            
        # 3. Download if missing
        self.log(f"Downloading {filename}...")
        try:
            url = "https://github.com/akanametov/yolo-face/releases/download/v0.0.0/yolov8n-face.pt"
            response = requests.get(url, stream=True)
            if response.status_code == 200:
                with open(filename, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=1024):
                        if chunk: f.write(chunk)
                return filename
        except Exception:
            pass
        return "yolov8n.pt" # Fallback

    def resolve_object_name(self, name):
        return self.resolver.resolve(name)

    def get_coords(self, box):
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        return int(x1), int(y1), int(x2-x1), int(y2-y1)

    def rotate_image(self, image, angle):
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        cos, sin = abs(M[0, 0]), abs(M[0, 1])
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))
        M[0, 2] += (new_w / 2) - center[0]
        M[1, 2] += (new_h / 2) - center[1]
        
        rotated = cv2.warpAffine(image, M, (new_w, new_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0))
        return rotated

    def overlay_glasses_advanced(self, frame, kpts, box):
        if kpts is None or len(kpts) < 2:
            x, y, w, h = box
            return self.overlay_glasses_simple(frame, x, y, w, h)

        le = kpts[0]
        re = kpts[1]
        
        if len(le) > 2 and (le[2] < 0.5 or re[2] < 0.5):
             x, y, w, h = box
             return self.overlay_glasses_simple(frame, x, y, w, h)

        lx, ly = int(le[0]), int(le[1])
        rx, ry = int(re[0]), int(re[1])

        dy = ry - ly
        dx = rx - lx
        eye_distance = math.sqrt(dx**2 + dy**2)
        angle = math.degrees(math.atan2(dy, dx))
        
        scale_mult = 2.5 
        g_w = int(eye_distance * scale_mult)
        aspect = self.glasses_img.shape[0] / self.glasses_img.shape[1]
        g_h = int(g_w * aspect)

        glasses_resized = cv2.resize(self.glasses_img, (g_w, g_h))
        glasses_rotated = self.rotate_image(glasses_resized, -angle)

        center_x = (lx + rx) // 2
        center_y = (ly + ry) // 2
        
        gh_rot, gw_rot = glasses_rotated.shape[:2]
        
        x1 = center_x - (gw_rot // 2)
        y1 = center_y - (gh_rot // 2)
        
        h_frm, w_frm = frame.shape[:2]
        if x1 < 0 or y1 < 0 or x1 + gw_rot > w_frm or y1 + gh_rot > h_frm:
             return frame

        alpha_s = glasses_rotated[:, :, 3] / 255.0
        alpha_l = 1.0 - alpha_s

        roi = frame[y1:y1+gh_rot, x1:x1+gw_rot]
        
        if roi.shape[:2] != glasses_rotated.shape[:2]:
            return frame

        for c in range(0, 3):
            roi[:, :, c] = (alpha_s * glasses_rotated[:, :, c] +
                            alpha_l * roi[:, :, c])
            
        frame[y1:y1+gh_rot, x1:x1+gw_rot] = roi
        return frame

    def overlay_glasses_simple(self, frame, x, y, w, h):
        scale_factor = 1.0
        g_w = int(w * scale_factor)
        aspect = self.glasses_img.shape[0] / self.glasses_img.shape[1]
        g_h = int(g_w * aspect)
        
        if g_w <= 0 or g_h <= 0: return frame
        glasses_resized = cv2.resize(self.glasses_img, (g_w, g_h))
        pos_x = int((x + w/2) - (g_w/2))
        pos_y = int((y + h/2.5) - (g_h/2)) # Lowered to h/2.5 as requested previously

        y1, y2 = max(0, pos_y), min(frame.shape[0], pos_y + g_h)
        x1, x2 = max(0, pos_x), min(frame.shape[1], pos_x + g_w)
        y1o, y2o = max(0, -pos_y), min(g_h, frame.shape[0] - pos_y)
        x1o, x2o = max(0, -pos_x), min(g_w, frame.shape[1] - pos_x)

        if y1 >= y2 or x1 >= x2 or y1o >= y2o or x1o >= x2o: return frame

        alpha_s = glasses_resized[y1o:y2o, x1o:x2o, 3] / 255.0
        alpha_l = 1.0 - alpha_s
        for c in range(0, 3):
            frame[y1:y2, x1:x2, c] = (alpha_s * glasses_resized[y1o:y2o, x1o:x2o, c] +
                                      alpha_l * frame[y1:y2, x1:x2, c])
        return frame

    def process_frame(self, frame, active_objects, patch_active):
        results_face = self.model_face(frame, verbose=False, conf=0.5)
        
        results_obj = None
        if active_objects:
            results_obj = self.model_objects(frame, verbose=False, conf=0.4)

        annotated_frame = frame.copy()

        for r in results_face:
            boxes = r.boxes
            keypoints = r.keypoints.data if r.keypoints is not None else None
            
            for i, box in enumerate(boxes):
                if int(box.cls[0]) == 0:
                    x, y, w, h = self.get_coords(box)
                    conf = float(box.conf[0])
                    
                    label_text = f"Face {conf:.0%}"
                    color = (0, 255, 0) 

                    if patch_active:
                        kpts = keypoints[i].cpu().numpy() if keypoints is not None else None
                        annotated_frame = self.overlay_glasses_advanced(annotated_frame, kpts, (x,y,w,h))
                        label_text = f"Panda {conf:.0%}" 
                        color = (0, 0, 255)

                    cv2.rectangle(annotated_frame, (x, y), (x+w, y+h), color, 2)
                    (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                    cv2.rectangle(annotated_frame, (x, y - 30), (x + tw, y), color, -1)
                    cv2.putText(annotated_frame, label_text, (x, y - 8), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        if results_obj:
            for r in results_obj:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    if cls_id < len(self.model_objects.names):
                        label_name = self.model_objects.names[cls_id].lower()
                        if label_name in active_objects:
                            x, y, w, h = self.get_coords(box)
                            conf = float(box.conf[0])
                            c_obj = (0, 255, 255) 
                            lbl = f"{label_name} {conf:.0%}"
                            cv2.rectangle(annotated_frame, (x, y), (x+w, y+h), c_obj, 2)
                            cv2.putText(annotated_frame, lbl, (x, y-10), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, c_obj, 2)

        return annotated_frame