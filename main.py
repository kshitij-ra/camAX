import numpy as np
import cv2 as cv
import mediapipe as mp
import time
import pyvirtualcam

from utils import (
    draw_landmarks_on_image_and_gestures,
    get_supported_resolutions_and_fps,
    overlay_png,
    rotate_fake3d,
    draw_glow
)

from gestures import get_left_right_hands, detect_heart
from effects_manager import EffectManager
from effects import HeartEffect

MODE = "landmarker"   # "gesture" or "landmarker"

if MODE == "gesture":
    from input.gesture_recognizer import HandInput
else:
    from input.hand_landmarker import HandInput

hand_input = HandInput()
manager = EffectManager(cooldown=100)

cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

# modes = get_supported_resolutions_and_fps(cap)

# print("Select camera mode:")
# for i, m in enumerate(modes):
#     print(f'{i}: {m["width"]}x{m["height"]} @ {m["fps"]} FPS')

# mode_id = int(input("Enter mode number: "))

# cap.set(cv.CAP_PROP_FRAME_WIDTH, modes[mode_id]["width"])
# cap.set(cv.CAP_PROP_FRAME_HEIGHT, modes[mode_id]["height"])
# cap.set(cv.CAP_PROP_FPS, modes[mode_id]["fps"])

cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)

print(
    "Using:",
    cap.get(cv.CAP_PROP_FRAME_WIDTH),
    cap.get(cv.CAP_PROP_FRAME_HEIGHT),
    cap.get(cv.CAP_PROP_FPS)
)

virtual_cam = pyvirtualcam.Camera(
    width=1280,
    height=720,
    fps=30,
    # print_fps=True
)

print("Virtual cam started:", virtual_cam.device)

fps_time = time.time()
fps_counter = 0
fps_display = 0

last_ts = 0

while True:

    ret, frame = cap.read()
    if not ret:
        break

    ts_now = int(time.monotonic() * 1000)
    if ts_now <= last_ts:
        ts_now = last_ts + 1
    last_ts = ts_now

    image = cv.flip(frame, 1)
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=image
    )

    hand_input.process(mp_image, ts_now)
    output = image.copy()
    hand_result = hand_input.latest_result

    if hand_result:
        #---------------------------------------------------------------
        # output = draw_landmarks_on_image_and_gestures(output, result)
        left_hand, right_hand = get_left_right_hands(hand_result, True)

        if left_hand and right_hand:
            if detect_heart(left_hand, right_hand):
                effect_x = int(
                    output.shape[1] *
                    (right_hand[8].x + left_hand[8].x) / 2
                )

                effect_y = int(
                    output.shape[0] *
                    (right_hand[8].y + left_hand[4].y) / 2
                )

                depth = (left_hand[8].z + right_hand[8].z) / 2

                manager.start_effect(
                    HeartEffect(effect_x, effect_y, depth),
                    ts_now
                )
        elif left_hand or right_hand:
            #detect other effects and call them
            pass

        manager.update()
        manager.draw(output)

    fps_counter += 1
    now = time.time()

    if now - fps_time >= 1.0:
        fps_display = fps_counter / (now - fps_time)
        fps_counter = 0
        fps_time = now

    #---------------------------------------------------------------
    # cv.putText(
    #     output,
    #     f"FPS: {int(fps_display)}",
    #     (20, 40),
    #     cv.FONT_HERSHEY_SIMPLEX,
    #     1,
    #     (0, 255, 0),
    #     2
    # )
    output = cv.flip(output, 1)
    virtual_cam.send(output)
    virtual_cam.sleep_until_next_frame()
    #---------------------------------------------------------------
    # cv.namedWindow("camAX", cv.WINDOW_NORMAL)
    # cv.imshow("camAX", cv.cvtColor(output, cv.COLOR_RGB2BGR))


    if cv.waitKey(1) == ord("q"):
        break

hand_input.close()
cap.release()
cv.destroyAllWindows()
