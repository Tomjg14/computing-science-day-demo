from PIL import Image, ImageDraw
import numpy as np
import os

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
        # Prioritize PNG as it's more likely to hold legitimate transparency data
        for filename in ["glasses.png", "patch.png", "glasses.jpg"]:
            if os.path.exists(filename):
                try:
                    print(f"[PATCH] Loading custom patch: {filename}")
                    # Always convert to RGBA (Red, Green, Blue, Alpha)
                    img = Image.open(filename).convert("RGBA")
                    arr = np.array(img)

                    # --- Automatic White-to-Transparent Hack ---
                    # Many downloaded images have a solid white background instead of real transparency.
                    # We find pixels that are pure white (RGB 255,255,255) and force their Alpha to 0.
                    
                    # Create a True/False mask where R, G, AND B channels are all 255
                    white_mask = (arr[:, :, 0] == 255) & \
                                 (arr[:, :, 1] == 255) & \
                                 (arr[:, :, 2] == 255)
                    
                    # Set the Alpha channel (index 3) of those white pixels to 0 (fully transparent)
                    arr[white_mask, 3] = 0
                    
                    return arr
                except Exception as e:
                    print(f"[PATCH] Error loading file, falling back: {e}")

        # 2. Fallback: Generate Pixel Art
        print("[PATCH] No custom file found. Generating default pixel art.")
        img = Image.new('RGBA', (300, 100), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        
        # Cyberpunk colors
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