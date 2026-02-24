import cv2
import torch
import os
import requests
import numpy as np
import math
import difflib
from ultralytics import YOLO
import config  # <--- IMPORT CONFIG
from patch_generator import PatchGenerator
from utils import resource_path

try:
    import face_recognition
    HAS_FACE_REC = True
except ImportError:
    HAS_FACE_REC = False
    print("[WARNING] 'face_recognition' library not found. Recognition disabled.")

class SmartResolver:
    def __init__(self, model_names):
        self.names = model_names 
        self.valid_list = [v.lower() for v in self.names.values()]
        
        # Synonyms map
        self.aliases = {
            "mobile": "cell phone", "phone": "cell phone", "iphone": "cell phone",
            "mug": "cup", "coffee": "cup",
            "coke": "bottle", "water": "bottle", "beer": "bottle",
            "screen": "tv", "monitor": "tv", "pc": "laptop", "macbook": "laptop",
            "controller": "remote", "man": "person", "woman": "person", "bike": "bicycle"
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
        self.HAS_FACE_REC = HAS_FACE_REC
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.status_callback = status_callback
        
        self.log(f"Loading Models on {self.device.upper()}...")
        
        # 1. Load Face Model
        face_path = self.get_best_face_model()
        self.log(f"Loading Face Model: {face_path}")
        self.model_face = YOLO(face_path) 
        
        # 2. Load Object Model
        obj_path = resource_path(config.MODEL_OBJ_FILENAME)
        if not os.path.exists(obj_path):
            obj_path = config.MODEL_OBJ_FILENAME
            
        self.log(f"Loading Object Model: {obj_path}")
        self.model_objects = YOLO(obj_path) 
        
        self.resolver = SmartResolver(self.model_objects.names)
        self.glasses_img = PatchGenerator.get_glasses_mask()
        
        # Load QR Code Overlay
        self.qr_img = None
        qr_path = resource_path(config.QR_CODE_FILENAME)
        if os.path.exists(qr_path):
            img = cv2.imread(qr_path, cv2.IMREAD_UNCHANGED)
            if img is not None:
                h, w = img.shape[:2]
                scale = config.QR_CODE_WIDTH / w
                self.qr_img = cv2.resize(img, (config.QR_CODE_WIDTH, int(h * scale)))
                self.log(f"Loaded QR Code: {config.QR_CODE_FILENAME}")

        self.log("Models Loaded.")
        
        # Face Recognition Data
        self.known_face_encodings = []
        self.known_face_names = []
        if HAS_FACE_REC: self.load_known_faces()

    def log(self, msg):
        if self.status_callback: self.status_callback(msg)
        print(f"[ENGINE] {msg}")

    def get_best_face_model(self):
        filename = config.MODEL_FACE_FILENAME
        
        # Check bundled/local
        bundled_path = resource_path(filename)
        if os.path.exists(bundled_path) and os.path.getsize(bundled_path) > 1000000:
            return bundled_path

        # Download
        self.log(f"Downloading {filename}...")
        try:
            response = requests.get(config.MODEL_FACE_URL, stream=True)
            if response.status_code == 200:
                with open(filename, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=1024):
                        if chunk: f.write(chunk)
                return filename
        except Exception:
            pass
        return config.MODEL_OBJ_FILENAME # Fallback

    def load_known_faces(self):
        if not os.path.exists(config.KNOWN_FACES_DIR):
            os.makedirs(config.KNOWN_FACES_DIR)
            return

        self.log("Loading Known Faces...")
        count = 0
        for filename in os.listdir(config.KNOWN_FACES_DIR):
            if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
                path = os.path.join(config.KNOWN_FACES_DIR, filename)
                try:
                    image = face_recognition.load_image_file(path)
                    encodings = face_recognition.face_encodings(image)
                    if encodings:
                        self.known_face_encodings.append(encodings[0])
                        name = os.path.splitext(filename)[0]
                        self.known_face_names.append(name)
                        count += 1
                except Exception as e:
                    print(f"Error loading {filename}: {e}")
        self.log(f"Loaded {count} known faces.")

    def register_face(self, frame, box, name):
        if not HAS_FACE_REC: return False, "Library missing"
        
        x, y, w, h = box
        # Ensure crop is valid
        h_frm, w_frm = frame.shape[:2]
        x, y = max(0, x), max(0, y)
        w, h = min(w, w_frm - x), min(h, h_frm - y)
        
        if w < 20 or h < 20: return False, "Face too small"

        face_img = frame[y:y+h, x:x+w]
        rgb_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        
        encodings = face_recognition.face_encodings(rgb_face)
        if not encodings:
            # Fallback: Force the YOLO box as the face location if dlib misses it
            h_img, w_img = rgb_face.shape[:2]
            encodings = face_recognition.face_encodings(rgb_face, known_face_locations=[(0, w_img, h_img, 0)])

        if not encodings: return False, "No face features found"
        
        # Save to disk
        if not os.path.exists(config.KNOWN_FACES_DIR): os.makedirs(config.KNOWN_FACES_DIR)
        
        clean_name = "".join(c for c in name if c.isalnum() or c in (' ', '-', '_')).strip()
        if not clean_name: clean_name = "unknown"
        
        save_path = os.path.join(config.KNOWN_FACES_DIR, f"{clean_name}.jpg")
        cv2.imwrite(save_path, face_img)
        
        self.known_face_encodings.append(encodings[0])
        self.known_face_names.append(clean_name)
        return True, f"Saved {clean_name}"

    def remove_face(self, name):
        if name in self.known_face_names:
            idx = self.known_face_names.index(name)
            self.known_face_names.pop(idx)
            self.known_face_encodings.pop(idx)
            
            # Delete file
            file_path = os.path.join(config.KNOWN_FACES_DIR, f"{name}.jpg")
            if os.path.exists(file_path):
                os.remove(file_path)
            return True
        return False

    def clear_known_faces(self):
        self.known_face_names.clear()
        self.known_face_encodings.clear()
        if os.path.exists(config.KNOWN_FACES_DIR):
            for f in os.listdir(config.KNOWN_FACES_DIR):
                os.remove(os.path.join(config.KNOWN_FACES_DIR, f))

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
        return cv2.warpAffine(image, M, (new_w, new_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0))

    def overlay_glasses_advanced(self, frame, kpts, box):
        if kpts is None or len(kpts) < 2:
            x, y, w, h = box
            return self.overlay_glasses_simple(frame, x, y, w, h)

        le, re = kpts[0], kpts[1]
        if len(le) > 2 and (le[2] < 0.5 or re[2] < 0.5):
             x, y, w, h = box
             return self.overlay_glasses_simple(frame, x, y, w, h)

        lx, ly = int(le[0]), int(le[1])
        rx, ry = int(re[0]), int(re[1])

        dy = ry - ly
        dx = rx - lx
        eye_dist = math.sqrt(dx**2 + dy**2)
        angle = math.degrees(math.atan2(dy, dx))
        
        # Use Config Scale
        g_w = int(eye_dist * config.GLASSES_SCALE)
        aspect = self.glasses_img.shape[0] / self.glasses_img.shape[1]
        g_h = int(g_w * aspect)

        glasses_rot = self.rotate_image(cv2.resize(self.glasses_img, (g_w, g_h)), -angle)
        
        center_x, center_y = (lx + rx) // 2, (ly + ry) // 2
        gh_rot, gw_rot = glasses_rot.shape[:2]
        
        x1 = center_x - (gw_rot // 2)
        y1 = center_y - (gh_rot // 2)
        
        h_frm, w_frm = frame.shape[:2]
        if x1 < 0 or y1 < 0 or x1 + gw_rot > w_frm or y1 + gh_rot > h_frm: return frame

        alpha_s = glasses_rot[:, :, 3] / 255.0
        roi = frame[y1:y1+gh_rot, x1:x1+gw_rot]
        if roi.shape[:2] != glasses_rot.shape[:2]: return frame

        for c in range(0, 3):
            roi[:, :, c] = (alpha_s * glasses_rot[:, :, c] + (1.0 - alpha_s) * roi[:, :, c])
            
        frame[y1:y1+gh_rot, x1:x1+gw_rot] = roi
        return frame

    def overlay_glasses_simple(self, frame, x, y, w, h):
        scale = 1.0
        g_w = int(w * scale)
        g_h = int(g_w * (self.glasses_img.shape[0] / self.glasses_img.shape[1]))
        if g_w <= 0 or g_h <= 0: return frame
        
        glasses_rez = cv2.resize(self.glasses_img, (g_w, g_h))
        pos_x = int((x + w/2) - (g_w/2))
        pos_y = int((y + h/2.5) - (g_h/2))

        y1, y2 = max(0, pos_y), min(frame.shape[0], pos_y + g_h)
        x1, x2 = max(0, pos_x), min(frame.shape[1], pos_x + g_w)
        y1o, y2o = max(0, -pos_y), min(g_h, frame.shape[0] - pos_y)
        x1o, x2o = max(0, -pos_x), min(g_w, frame.shape[1] - pos_x)

        if y1 >= y2 or x1 >= x2: return frame

        alpha_s = glasses_rez[y1o:y2o, x1o:x2o, 3] / 255.0
        for c in range(0, 3):
            frame[y1:y2, x1:x2, c] = (alpha_s * glasses_rez[y1o:y2o, x1o:x2o, c] + (1.0 - alpha_s) * frame[y1:y2, x1:x2, c])
        return frame

    def identify_face(self, frame, x, y, w, h):
        if not HAS_FACE_REC or not self.known_face_encodings:
            return None, 0.0

        face_img = frame[y:y+h, x:x+w]
        if face_img.size == 0: return None, 0.0
        
        rgb_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        try:
            # Try detection first, fallback to forced location
            encodings = face_recognition.face_encodings(rgb_face)
            if not encodings:
                h_img, w_img = rgb_face.shape[:2]
                encodings = face_recognition.face_encodings(rgb_face, known_face_locations=[(0, w_img, h_img, 0)])

            if encodings:
                matches = face_recognition.compare_faces(self.known_face_encodings, encodings[0], tolerance=config.FACE_MATCH_THRESHOLD)
                dists = face_recognition.face_distance(self.known_face_encodings, encodings[0])
                best_idx = np.argmin(dists)
                if matches[best_idx]:
                    return self.known_face_names[best_idx], 1.0 - dists[best_idx]
        except Exception: pass
        return None, 0.0

    def process_frame(self, frame, active_objects, patch_active):
        results_face = self.model_face(frame, verbose=False, conf=config.CONF_THRESHOLD_FACE)
        
        results_obj = None
        if active_objects:
            results_obj = self.model_objects(frame, verbose=False, conf=config.CONF_THRESHOLD_OBJ)

        annotated = frame.copy()

        # Draw Faces
        for r in results_face:
            boxes = r.boxes
            kpts = r.keypoints.data if r.keypoints is not None else None
            
            for i, box in enumerate(boxes):
                if int(box.cls[0]) == 0:
                    x, y, w, h = self.get_coords(box)
                    conf = float(box.conf[0])
                    
                    label = f"Face {conf:.0%}"
                    color = config.COLOR_FACE_NORMAL

                    # Attempt Recognition
                    rec_name, rec_conf = self.identify_face(frame, x, y, w, h)
                    if rec_name:
                        label = f"{rec_name} {rec_conf:.0%}"

                    if patch_active:
                        kp = kpts[i].cpu().numpy() if kpts is not None else None
                        annotated = self.overlay_glasses_advanced(annotated, kp, (x,y,w,h))
                        label = f"Panda {conf:.0%}" 
                        color = config.COLOR_FACE_PANDA

                    cv2.rectangle(annotated, (x, y), (x+w, y+h), color, 2)
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                    cv2.rectangle(annotated, (x, y-30), (x+tw, y), color, -1)
                    cv2.putText(annotated, label, (x, y-8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Draw Objects
        if results_obj:
            for r in results_obj:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    if cls_id < len(self.model_objects.names):
                        name = self.model_objects.names[cls_id].lower()
                        if name in active_objects:
                            x, y, w, h = self.get_coords(box)
                            conf = float(box.conf[0])
                            color = config.COLOR_OBJECT
                            lbl = f"{name} {conf:.0%}"
                            cv2.rectangle(annotated, (x, y), (x+w, y+h), color, 2)
                            cv2.putText(annotated, lbl, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Draw QR Code Overlay
        if self.qr_img is not None:
            qh, qw = self.qr_img.shape[:2]
            qx, qy = config.QR_CODE_POS
            h_frm, w_frm = annotated.shape[:2]
            
            if qy + qh < h_frm and qx + qw < w_frm:
                # Handle Transparency (4 channels) vs Standard (3 channels)
                if len(self.qr_img.shape) > 2 and self.qr_img.shape[2] == 4:
                    alpha = self.qr_img[:, :, 3] / 255.0
                    roi = annotated[qy:qy+qh, qx:qx+qw]
                    for c in range(3):
                        roi[:, :, c] = (alpha * self.qr_img[:, :, c] + (1.0 - alpha) * roi[:, :, c])
                    annotated[qy:qy+qh, qx:qx+qw] = roi
                else:
                    annotated[qy:qy+qh, qx:qx+qw] = self.qr_img[:, :, :3]

        return annotated