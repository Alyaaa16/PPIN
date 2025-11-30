import numpy as np
import cv2
from PIL import Image

class ImageAspect:
    """
    Replacement for missing image_aspect module.
    Supports:
    - change_aspect_rate()
    - past_background()
    - PIL2ndarray()
    - save_rate()
    """
    def __init__(self, img, target_h, target_w):
        # Convert PIL → numpy if needed
        if isinstance(img, Image.Image):
            img = np.array(img)

        self.original = img                      # original image (H,W,3)
        self.orig_h, self.orig_w = img.shape[:2]
        self.target_h = target_h
        self.target_w = target_w

        self.scale = None
        self.offset = None
        self._resized_canvas = None

    def change_aspect_rate(self):
        """Resize with aspect ratio preserved."""
        scale = min(self.target_h / self.orig_h,
                    self.target_w / self.orig_w)
        new_h = int(self.orig_h * scale)
        new_w = int(self.orig_w * scale)

        # resize
        resized = cv2.resize(self.original, (new_w, new_h))

        # store internal values
        self.scale = scale
        self.new_h = new_h
        self.new_w = new_w
        self._resized = resized
        return self

    def past_background(self):
        """Paste resized image onto black background canvas."""
        canvas = np.zeros((self.target_h, self.target_w, 3),
                          dtype=self._resized.dtype)

        offset_x = (self.target_w - self.new_w) // 2
        offset_y = (self.target_h - self.new_h) // 2

        canvas[offset_y:offset_y+self.new_h,
               offset_x:offset_x+self.new_w] = self._resized

        self.offset = (offset_x, offset_y)
        self._resized_canvas = canvas
        return self

    def PIL2ndarray(self):
        """Return final resized numpy image."""
        return self._resized_canvas

    def save_rate(self):
        """Return (scale factor, offset tuple)."""
        if self.scale is None or self.offset is None:
            raise ValueError("Call change_aspect_rate().past_background() first.")
        return self.scale, self.offset
