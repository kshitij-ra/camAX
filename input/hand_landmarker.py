import mediapipe as mp

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


class HandInput:
    def __init__(self, model_path="models/hand_landmarker.task"):
        self.latest_result = None

        self.options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.LIVE_STREAM,
            result_callback=self._callback,
            num_hands=2
        )

        self.landmarker = HandLandmarker.create_from_options(self.options)

    def _callback(self, result, output_image, timestamp_ms):
        self.latest_result = result

    def process(self, mp_image, timestamp):
        self.landmarker.detect_async(mp_image, timestamp)

    def close(self):
        self.landmarker.close()
