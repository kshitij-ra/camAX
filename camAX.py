import numpy as np
import cv2 as cv
import mediapipe as mp
import time

BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
VisionRunningMode = mp.tasks.vision.RunningMode

mp_hands = mp.tasks.vision.HandLandmarksConnections
mp_drawing = mp.tasks.vision.drawing_utils
mp_drawing_styles = mp.tasks.vision.drawing_styles

latest_result = None

MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green

effect_x = None
effect_y = None

effect_in_progress = False

hearts = []
last_effect_time = 0
EFFECT_INTERVAL = 100  # ms

heart_png = cv.imread("heart.png", cv.IMREAD_UNCHANGED)  # RGBA
heart_png = cv.resize(heart_png,(64,64))
heart_png = cv.cvtColor(heart_png, cv.COLOR_RGBA2BGRA)


def get_left_right_hands(detection_result,flip=False):
    left_hand = None
    right_hand = None

    for i in range(len(detection_result.hand_landmarks)):
        label = detection_result.handedness[i][0].category_name
        hand = detection_result.hand_landmarks[i]

        if label == "Left":
            left_hand = hand
        elif label == "Right":
            right_hand = hand

    return (right_hand, left_hand) if flip else (left_hand, right_hand)

def detect_heart(hand_left, hand_right):
    THUMB_TIP = 4
    INDEX_TIP = 8

    left_thumb = hand_left[THUMB_TIP]
    right_thumb = hand_right[THUMB_TIP]

    left_index = hand_left[INDEX_TIP]
    right_index = hand_right[INDEX_TIP]

    thumb_dist = np.linalg.norm(
        np.array([left_thumb.x, left_thumb.y]) -
        np.array([right_thumb.x, right_thumb.y])
    )

    index_dist = np.linalg.norm(
        np.array([left_index.x, left_index.y]) -
        np.array([right_index.x, right_index.y])
    )

    return thumb_dist < 0.05 and index_dist < 0.05

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

def print_result(result: GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
    # print('hand landmarker result: {}'.format(result))
    global latest_result
    latest_result = result

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

def draw_landmarks_on_image_and_gestures(rgb_image, detection_result):
    annotated_image = np.copy(rgb_image)

    if not detection_result.hand_landmarks:
        return annotated_image

    for idx in range(len(detection_result.hand_landmarks)):
        hand_landmarks = detection_result.hand_landmarks[idx]
        handedness = detection_result.handedness[idx]
        gesture = detection_result.gestures[idx]

        # Draw landmarks
        mp_drawing.draw_landmarks(
            annotated_image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style()
        )

        # Position for text
        h, w, _ = annotated_image.shape
        xs = [lm.x for lm in hand_landmarks]
        ys = [lm.y for lm in hand_landmarks]

        text_x = int(min(xs) * w)
        text_y = int(min(ys) * h) - MARGIN

        gesture_name = gesture[0].category_name if gesture else "None"

        cv.putText(
            annotated_image,
            f"{handedness[0].category_name} {gesture_name}",
            (text_x, text_y),
            cv.FONT_HERSHEY_DUPLEX,
            FONT_SIZE,
            HANDEDNESS_TEXT_COLOR,
            FONT_THICKNESS,
            cv.LINE_AA
        )

    return annotated_image


options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path='gesture_recognizer.task'),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result,
    num_hands=2
    )


cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()
#show resolution
# print(cap.get(cv.CAP_PROP_FRAME_WIDTH), cap.get(cv.CAP_PROP_FRAME_HEIGHT))
#set resolution
# resolution = (1920,1080)
# cap.set(cv.CAP_PROP_FRAME_WIDTH, resolution[0])
# cap.set(cv.CAP_PROP_FRAME_HEIGHT, resolution[1])
# print(cap.get(cv.CAP_PROP_FRAME_WIDTH), cap.get(cv.CAP_PROP_FRAME_HEIGHT))

with GestureRecognizer.create_from_options(options) as recognizer:
    old_time = 0
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        # create timestamp
        frame_timestamp_now = int(time.monotonic() * 1000)
        # time should be monotonically increasing
        if frame_timestamp_now <= old_time:
            frame_timestamp_now = old_time + 1
        old_time = frame_timestamp_now
        # operations on the frame come here
        image = cv.flip(frame, 1)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
        # recognize async for live stream
        recognizer.recognize_async(mp_image, frame_timestamp_now)

        if latest_result is not None:
            # draw landmarks on image
            annotated_image = draw_landmarks_on_image_and_gestures(image, latest_result)
            # annotated_image = image
            
            # detection for heart gesture
            left_hand, right_hand = get_left_right_hands(latest_result,True)
            if left_hand is not None and right_hand is not None:
                if detect_heart(left_hand, right_hand):
                    cv.putText(
                        annotated_image,
                        "HEART GESTURE",
                        (50, 80),
                        cv.FONT_HERSHEY_SIMPLEX,
                        1.5,
                        (0, 0, 255),
                        3
                    )

                    # emit hearts
                    #prevent spamming
                    if frame_timestamp_now - last_effect_time > EFFECT_INTERVAL:
                        last_effect_time = frame_timestamp_now

                        # effect position
                        effect_x = int(annotated_image.shape[1]*(right_hand[8].x + left_hand[8].x)/2)
                        effect_y = int(annotated_image.shape[0]*(right_hand[8].y + left_hand[4].y)/2)
                        depth = (left_hand[8].z + right_hand[8].z) / 2
                        base_scale = np.clip(1.0 - depth * 2, 0.4, 1.6)

                        for _ in range(1):
                            hearts.append({
                                "x": effect_x + np.random.randint(-50, 50),
                                "y": effect_y + np.random.randint(-30, 30),
                                "vx": np.random.uniform(-0.5, 1),
                                "vy": np.random.uniform(1.5, 3.5),
                                "life": 90 - np.random.randint(0, 15),
                                "maxlife": 90,
                                "size": np.random.randint(10,16),
                                "angle": np.random.uniform(0, np.pi),
                                "spin": np.random.uniform(0.05, 0.12),
                                "scale": np.random.uniform(0.1, 0.15),
                                "targetscale": base_scale * np.random.uniform(0.8, 1.2)
                            })
            alive_hearts = []
            for heart in hearts:
                # cv.circle(
                #     annotated_image,
                #     (int(heart["x"]), int(heart["y"])),
                #     heart["size"],
                #     (0, 0, 255),
                #     -1
                # )
                sw = max(1, int(heart_png.shape[1] * heart["scale"]))
                sh = max(1, int(heart_png.shape[0] * heart["scale"]))
                scaled_png = cv.resize(heart_png, (sw, sh))
                sprite = rotate_fake3d(scaled_png, heart["angle"])
                brightness = 0.6 + 0.4 * abs(np.cos(heart["angle"]))
                sprite[:, :, :3] = np.clip(sprite[:, :, :3] * brightness, 0, 255)
                h, w = sprite.shape[:2]
                alpha = heart["life"] / heart["maxlife"]
                # draw_glow(annotated_image, sprite, int(heart["x"] - w // 2), int(heart["y"] - h // 2))
                overlay_png(annotated_image, sprite, int(heart["x"] - w // 2), int(heart["y"] - h // 2), alpha)
                heart["x"] += heart["vx"] + np.random.uniform(-0.1,0.1)
                heart["vy"] += 0.05
                heart["vy"] = min(heart["vy"], 5)
                heart["y"] -= heart["vy"]
                heart["scale"] += (heart["targetscale"] - heart["scale"]) * 0.05
                heart["angle"] += heart["spin"]
                heart["life"] -= 1

                if heart["life"] > 0:
                    alive_hearts.append(heart)
            
            hearts = alive_hearts
        else:
            annotated_image = image
        
        # Display the resulting frame
        cv.imshow('frame', cv.cvtColor(annotated_image, cv.COLOR_RGB2BGR))
        if cv.waitKey(1) == ord('q'):
            break

# Release the capture
cap.release()
cv.destroyAllWindows()