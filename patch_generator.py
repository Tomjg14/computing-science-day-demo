from PIL import Image, ImageDraw
import numpy as np
import os
from utils import resource_path  # Import the helper

class PatchGenerator:
    """
    Helper to generate or load the glasses mask.
    Prioritizes loading 'glasses.png' or 'glasses.jpg' from disk.
    If a custom file is loaded, it automatically converts pure white pixels to transparent.
    Falls back to generated pixel art if files are missing.
    """
    @staticmethod
    def get_glasses_mask():
        # 1. Try Loading from File
        # We wrap filenames in resource_path() to find them in the .exe
        filenames = ["glasses.png", "patch.png", "glasses.jpg"]
        
        for name in filenames:
            full_path = resource_path(name)
            if os.path.exists(full_path):
                try:
                    print(f"[PATCH] Loading custom patch: {full_path}")
                    # Always convert to RGBA
                    img = Image.open(full_path).convert("RGBA")
                    arr = np.array(img)

                    # --- Automatic White-to-Transparent Hack ---
                    white_mask = (arr[:, :, 0] == 255) & \
                                 (arr[:, :, 1] == 255) & \
                                 (arr[:, :, 2] == 255)
                    arr[white_mask, 3] = 0
                    
                    return arr
                except Exception as e:
                    print(f"[PATCH] Error loading file: {e}")

        # 2. Fallback: Generate Pixel Art
        print("[PATCH] No custom file found. Generating default pixel art.")
        img = Image.new('RGBA', (300, 100), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        
        colors = [(0, 255, 255, 255), (255, 0, 255, 255)]
        
        # Left Lens
        draw.rectangle([20, 10, 130, 90], fill=(0, 0, 0, 255))
        for x in range(30, 120, 10):
            for y in range(20, 80, 10):
                draw.rectangle([x, y, x+5, y+5], fill=colors[(x+y)%2])

        # Right Lens
        draw.rectangle([170, 10, 280, 90], fill=(0, 0, 0, 255))
        for x in range(180, 270, 10):
            for y in range(20, 80, 10):
                draw.rectangle([x, y, x+5, y+5], fill=colors[(x+y)%2])
                
        # Bridge
        draw.rectangle([130, 40, 170, 50], fill=(0, 0, 0, 255))
        
        return np.array(img)