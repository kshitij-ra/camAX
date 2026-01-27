# all effects
from utils import overlay_png, rotate_fake3d
import numpy as np
import cv2 as cv

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class Effect:
    def update(self):
        pass

    def draw(self, frame):
        pass

    def alive(self):
        return True

class HeartEffect(Effect):
    def __init__(self,effect_x,effect_y,depth):
        self.x = effect_x
        self.y = effect_y
        self.depth = depth
        self.base_scale = np.clip(1.0 - depth * 2, 0.4, 1.6)
        self.hearts = []
        self.heart_png = cv.imread("assets/heart.png", cv.IMREAD_UNCHANGED)  # RGBA
        self.heart_png = cv.resize(self.heart_png,(64,64))
        self.heart_png = cv.cvtColor(self.heart_png, cv.COLOR_RGBA2BGRA)

        for _ in range(1):
            self.hearts.append({
                "x": self.x + np.random.randint(-50, 50),
                "y": self.y + np.random.randint(-30, 30),
                "vx": np.random.uniform(-0.5, 1),
                "vy": np.random.uniform(1.5, 3.5),
                "life": 90 - np.random.randint(0, 15),
                "maxlife": 90,
                "size": np.random.randint(10,16),
                "angle": np.random.uniform(0, np.pi),
                "spin": np.random.uniform(0.05, 0.12),
                "scale": np.random.uniform(0.1, 0.15),
                "targetscale": self.base_scale * np.random.uniform(0.8, 1.2)
            })

    def update(self):
        alive_hearts = []
        for heart in self.hearts:
            heart["x"] += heart["vx"] + np.random.uniform(-0.1,0.1)
            heart["vy"] += 0.05
            heart["vy"] = min(heart["vy"], 5)
            heart["y"] -= heart["vy"]
            heart["scale"] += (heart["targetscale"] - heart["scale"]) * 0.05
            heart["angle"] += heart["spin"]
            heart["life"] -= 1

            if heart["life"] > 0:
                alive_hearts.append(heart)
        self.hearts = alive_hearts

    def draw(self,frame):
        for heart in self.hearts:
            sw = max(1, int(self.heart_png.shape[1] * heart["scale"]))
            sh = max(1, int(self.heart_png.shape[0] * heart["scale"]))
            scaled_png = cv.resize(self.heart_png, (sw, sh))
            sprite = rotate_fake3d(scaled_png, heart["angle"])
            brightness = 0.6 + 0.4 * abs(np.cos(heart["angle"]))
            sprite[:, :, :3] = np.clip(sprite[:, :, :3] * brightness, 0, 255)
            h, w = sprite.shape[:2]
            alpha = heart["life"] / heart["maxlife"]
            cx = int(heart["x"] - w // 2)
            cy = int(heart["y"] - h // 2) 
            overlay_png(frame, sprite, cx, cy, alpha)
        

    def alive(self):
        return len(self.hearts) > 0

class fireworksEffect(Effect):
    def __init__(self,effect_x,effect_y,depth):
        self.x = effect_x
        self.y = effect_y
        self.depth = depth

    def update(self):
        pass

    def draw(self,frame):
        pass

    def alive(self):
        return True

class lazershowEffect(Effect):
    def __init__(self,effect_x,effect_y,depth):
        self.x = effect_x
        self.y = effect_y
        self.depth = depth

    def update(self):
        pass

    def draw(self,frame):
        pass

    def alive(self):
        return True