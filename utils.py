import numpy as np
import cv2 as cv
import mediapipe as mp

mp_hands = mp.tasks.vision.HandLandmarksConnections
mp_drawing = mp.tasks.vision.drawing_utils
mp_drawing_styles = mp.tasks.vision.drawing_styles

def draw_glow(bg, sprite, x, y, intensity=1.8):
    h, w = sprite.shape[:2]

    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(bg.shape[1], x + w)
    y2 = min(bg.shape[0], y + h)

    if x1 >= x2 or y1 >= y2:
        return

    sx1 = x1 - x
    sy1 = y1 - y
    sx2 = sx1 + (x2 - x1)
    sy2 = sy1 + (y2 - y1)

    sprite_crop = sprite[sy1:sy2, sx1:sx2]
    glow = sprite_crop[:, :, :3].copy()
    alpha = sprite_crop[:, :, 3] / 255.0

    glow = np.clip(glow * intensity, 0, 255).astype(np.uint8)
    glow = cv.GaussianBlur(glow, (15,15), 0)

    for c in range(3):
        bg[y1:y2, x1:x2, c] = (
            bg[y1:y2, x1:x2, c] * (1 - alpha) +
            glow[:, :, c] * alpha
        )

def overlay_png(bg, png, x, y, alpha=1):
    h, w = png.shape[:2]

    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(bg.shape[1], x + w)
    y2 = min(bg.shape[0], y + h)

    if x1 >= x2 or y1 >= y2:
        return

    sx1 = x1 - x
    sy1 = y1 - y
    sx2 = sx1 + (x2 - x1)
    sy2 = sy1 + (y2 - y1)

    png_crop = png[sy1:sy2, sx1:sx2]

    rgb = png_crop[:, :, :3]
    a = (png_crop[:, :, 3] / 255.0) * alpha

    for c in range(3):
        bg[y1:y2, x1:x2, c] = (
            bg[y1:y2, x1:x2, c] * (1 - a) +
            rgb[:, :, c] * a
        )

def rotate_fake3d(png, angle):
    h, w = png.shape[:2]
    squash = 0.15 + 0.85 * abs(np.cos(angle))
    # squash = max(squash, 0.15)   # avoid disappearing
    new_w = max(1, int(w * squash))
    return cv.resize(png, (new_w, h))


MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green

def draw_landmarks_on_image_and_gestures(rgb_image, detection_result):
    annotated_image = rgb_image.copy()

    if not detection_result.hand_landmarks:
        return annotated_image

    has_gestures = hasattr(detection_result, "gestures")

    for idx in range(len(detection_result.hand_landmarks)):

        hand_landmarks = detection_result.hand_landmarks[idx]
        handedness = detection_result.handedness[idx]

        gesture_name = "None"

        if has_gestures:
            gesture_list = detection_result.gestures[idx]
            if gesture_list:
                gesture_name = gesture_list[0].category_name

        # Draw landmarks
        mp_drawing.draw_landmarks(
            annotated_image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style()
        )

        h, w, _ = annotated_image.shape
        xs = [lm.x for lm in hand_landmarks]
        ys = [lm.y for lm in hand_landmarks]

        text_x = int(min(xs) * w)
        text_y = int(min(ys) * h) - 10

        label = handedness[0].category_name

        if has_gestures:
            label += f" {gesture_name}"

        cv.putText(
            annotated_image,
            label,
            (text_x, text_y),
            cv.FONT_HERSHEY_DUPLEX,
            1,
            (88, 205, 54),
            1,
            cv.LINE_AA
        )

    return annotated_image


def get_supported_resolutions_and_fps(cap, try_fps=30):
    common_resolutions = [
        (320, 240),
        (640, 480),
        (800, 600),
        (1024, 768),
        (1280, 720),
        (1280, 800),
        (1366, 768),
        (1600, 900),
        (1920, 1080),
        (2560, 1440),
        (3840, 2160)
    ]

    supported = []

    for w, h in common_resolutions:
        # try setting resolution
        cap.set(cv.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv.CAP_PROP_FRAME_HEIGHT, h)

        # try setting fps
        cap.set(cv.CAP_PROP_FPS, try_fps)

        actual_w = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        actual_fps = cap.get(cv.CAP_PROP_FPS)

        if actual_w == w and actual_h == h:
            supported.append({
                "width": actual_w,
                "height": actual_h,
                "fps": round(actual_fps, 1)
            })

    return supported

