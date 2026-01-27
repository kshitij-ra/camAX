# detect custom gesture and helper functions
import numpy as np

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
