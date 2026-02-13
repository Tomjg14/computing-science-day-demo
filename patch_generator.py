from PIL import Image, ImageDraw
import numpy as np
import os
import config  # <--- IMPORT CONFIG
from utils import resource_path

class PatchGenerator:
    @staticmethod
    def get_glasses_mask():
        # 1. Try Loading from File (Using Config List)
        for name in config.PATCH_FILES:
            full_path = resource_path(name)
            if os.path.exists(full_path):
                try:
                    print(f"[PATCH] Loading custom patch: {full_path}")
                    img = Image.open(full_path).convert("RGBA")
                    arr = np.array(img)

                    # Auto-Transparency Hack (White -> Transparent)
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
        
        # Draw retro glasses
        draw.rectangle([20, 10, 130, 90], fill=(0, 0, 0, 255))
        for x in range(30, 120, 10):
            for y in range(20, 80, 10):
                draw.rectangle([x, y, x+5, y+5], fill=colors[(x+y)%2])
        draw.rectangle([170, 10, 280, 90], fill=(0, 0, 0, 255))
        for x in range(180, 270, 10):
            for y in range(20, 80, 10):
                draw.rectangle([x, y, x+5, y+5], fill=colors[(x+y)%2])
        draw.rectangle([130, 40, 170, 50], fill=(0, 0, 0, 255))
        
        return np.array(img)