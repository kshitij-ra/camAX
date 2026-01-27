import mediapipe as mp
import numpy as np
import cv2

BaseOptions = mp.tasks.BaseOptions
ImageSegmenter = mp.tasks.vision.ImageSegmenter
ImageSegmenterOptions = mp.tasks.vision.ImageSegmenterOptions
RunningMode = mp.tasks.vision.RunningMode

class PersonSegmenter:
    # lower smooth kernel or set to none if smoothening is heavy 
    def __init__(self, model_path="models/selfie_segmenter_landscape.tflite", threshold=0.2, smooth_kernel=15):
        self.latest_mask = None
        self.threshold = threshold
        self.smooth_kernel = smooth_kernel

        options = ImageSegmenterOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=RunningMode.LIVE_STREAM,
            output_category_mask=True,
            result_callback=self._callback
        )

        self.segmenter = ImageSegmenter.create_from_options(options)

    def _callback(self, result, output_image, timestamp_ms):
        if result.category_mask is not None:
            self.latest_mask = result.category_mask.numpy_view().copy()

    def process(self, mp_image, timestamp):

        self.segmenter.segment_async(mp_image, timestamp)

        if self.latest_mask is None:
            return None

        mask = self.latest_mask

        try:
            h, w = mp_image.numpy_view().shape[:2]
            if mask.shape[:2] != (h, w):
                return None
        except:
            pass

        mask = (mask > self.threshold).astype(np.uint8) * 255

        if self.smooth_kernel and self.smooth_kernel > 1:
            if self.smooth_kernel % 2 == 0:
                self.smooth_kernel += 1  # must be odd
            mask = cv2.GaussianBlur(mask, (self.smooth_kernel, self.smooth_kernel), 0)

        return mask

    def close(self):
        self.segmenter.close()
