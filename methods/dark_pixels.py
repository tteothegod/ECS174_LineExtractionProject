"""Sanity-check method (Role 1): "dark pixels are lines".

Not part of the experiment. It exists so evaluate.py can be tested before
Canny and the U-Net are ready, and as a copyable example of the predict()
contract for Roles 2 and 3. A real method should score far above this.
"""

import numpy as np


def predict(image, resolution):
    """Line probability = how dark the pixel is (1 = black, 0 = white)."""
    gray = image.mean(axis=2)
    return np.clip(1.0 - gray, 0.0, 1.0)
