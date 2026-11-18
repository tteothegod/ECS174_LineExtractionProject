"""Role 2: Canny edge detection baseline.

The classical method we compare the U-Net against. evaluate.py calls
predict(image, resolution) for every image:
  * image is ALREADY shrunk to the resolution (512 / 256 / 128 px)
  * we return a 0/1 line map with the SAME height and width (no resizing here)
evaluate.py then grows our map back to 512 x 512 and scores it.

Steps inside predict():
  1. color -> gray            (Canny works on brightness only)
  2. Gaussian blur            (remove tiny noise so it is not drawn as lines)
  3. Canny                    (find 1-pixel-wide edges)
  4. thicken (optional)       (artist lines are wider than 1 pixel)
"""

import cv2
import numpy as np

# Canny's output is already 0/1, so evaluate.py must not tune a threshold.
IS_BINARY = True

# Settings for each resolution, picked by tune_canny.py (best mean Dice on
# 300 validation images) and pasted here by hand.
#   sigma:  blur strength (0 = no blur)
#   low:    weak-edge threshold (edge strength; can go above 255)
#   high:   strong-edge threshold
#   dilate: how many pixels to thicken each line by (0 = keep 1 pixel wide)
BEST_PARAMS = {
    "full":    {"sigma": 0, "low": 400, "high": 500, "dilate": 1},   # val Dice 0.475
    "half":    {"sigma": 0, "low": 300, "high": 400, "dilate": 0},   # val Dice 0.321
    "quarter": {"sigma": 0, "low": 250, "high": 300, "dilate": 0},   # val Dice 0.255
}


def to_gray_uint8(image):
    """Color image (H x W x 3, float 0-1, RGB) -> gray image (H x W, uint8 0-255).

    cv2.Canny only accepts 8-bit images, so we convert in two steps:
      1. RGB -> gray with OpenCV's standard weights
         (gray = 0.299 R + 0.587 G + 0.114 B: our eyes see green as brightest)
      2. 0-1 floats -> 0-255 whole numbers
    """
    gray = cv2.cvtColor(image.astype(np.float32), cv2.COLOR_RGB2GRAY)
    return np.clip(np.round(gray * 255), 0, 255).astype(np.uint8)


def blur(gray, sigma):
    """Gaussian blur with strength sigma. sigma = 0 means no blur.

    Each pixel becomes a weighted average of its neighbors (closer neighbors
    count more). Tiny noise and texture get averaged away, while real edges
    (big brightness jumps) survive. ksize=(0, 0) lets OpenCV pick the window
    size from sigma (about 6 * sigma wide).
    """
    if sigma <= 0:
        return gray
    return cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma, sigmaY=sigma)


def detect_edges(gray, low, high):
    """Canny edge detection -> 0/1 map (uint8), lines 1 pixel wide.

    cv2.Canny does 4 things inside:
      1. Sobel filters measure how fast brightness changes in x and y
         -> gradient strength (how big the jump is) and direction at each pixel
      2. Non-maximum suppression: keep a pixel only if it is the strongest
         along its gradient direction, so thick blurry edges become 1 pixel wide
      3. Double threshold: strength >= high -> strong edge,
         low <= strength < high -> weak edge, below low -> thrown away
      4. Hysteresis: keep a weak edge only if it touches a strong edge
         (so faint parts of real lines survive, isolated noise does not)
    cv2.Canny returns 0 / 255; we divide by 255 to get 0 / 1.
    """
    edges = cv2.Canny(gray, threshold1=low, threshold2=high)
    return (edges // 255).astype(np.uint8)


def thicken(edges, dilate):
    """Widen every line by `dilate` pixels on each side. dilate = 0 means no change.

    Dilation slides a small round window (2 * dilate + 1 pixels wide) over the
    map: if ANY pixel under the window is a line, the center pixel becomes a
    line too. So a 1-pixel line becomes 3 pixels wide with dilate = 1, 5 with 2.
    Why: artist lines are wider than Canny's 1-pixel lines, and Canny draws two
    thin lines (one on each side of a dark stroke); widening merges the pair
    into one line on top of the artist's line.
    Same round window as metrics.py uses for its 1-pixel tolerance.
    """
    if dilate <= 0:
        return edges
    size = 2 * dilate + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    return cv2.dilate(edges, kernel)


def predict(image, resolution, sigma=None, low=None, high=None, dilate=None):
    """Color image -> 0/1 line map (H x W float32), same size as the input.

    image:      H x W x 3 float 0-1 RGB, ALREADY shrunk by evaluate.py
    resolution: "full", "half" or "quarter" (picks the row of BEST_PARAMS)
    sigma, low, high, dilate: optional overrides. evaluate.py never passes
        them (so BEST_PARAMS is used); tune_canny.py passes them to try
        other settings without editing this file.
    No resizing happens here: evaluate.py grows our output back to 512.
    """
    params = BEST_PARAMS[resolution]
    sigma = params["sigma"] if sigma is None else sigma
    low = params["low"] if low is None else low
    high = params["high"] if high is None else high
    dilate = params["dilate"] if dilate is None else dilate

    gray = to_gray_uint8(image)          # 1. color -> gray
    gray = blur(gray, sigma)             # 2. remove tiny noise
    edges = detect_edges(gray, low, high)  # 3. 1-pixel edges, 0/1
    edges = thicken(edges, dilate)       # 4. widen lines (optional)
    # evaluate.py expects floats in [0, 1]; ours are exactly 0.0 or 1.0
    return edges.astype(np.float32)


if __name__ == "__main__":
    # Self-check. Run from team_repo/:
    #   ANIME_DATA_ROOT=../data ../.venv/bin/python methods/canny.py
    # Runs predict() on one validation image at all 3 resolutions, checks the
    # output follows evaluate.py's rules, prints scores, saves a preview.
    import os
    import sys

    # Running this file directly puts methods/ on Python's search path, not
    # team_repo/, so `import data` would fail. Add team_repo/ (one folder up).
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, repo)
    from data import RESOLUTIONS, downsample_image, load_pair, load_splits, upsample_prob
    from metrics import all_scores

    image_id = load_splits()["val"][0]
    color, gt = load_pair(image_id)
    print("self-check on", image_id, "with BEST_PARAMS")

    def tile(img01, label):
        """0-1 image (gray or RGB) -> 256 x 256 BGR uint8 tile with a label."""
        img = (np.clip(img01, 0, 1) * 255).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR if img.ndim == 2 else cv2.COLOR_RGB2BGR)
        img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
        cv2.putText(img, label, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        return img

    tiles = [tile(color, "color")]
    for res, size in RESOLUTIONS.items():
        small = downsample_image(color, size)
        out = predict(small, res)
        # the same rules evaluate.py checks
        assert out.shape == small.shape[:2], "wrong shape %s" % (out.shape,)
        assert out.dtype == np.float32, "wrong dtype %s" % out.dtype
        assert set(np.unique(out)) <= {0.0, 1.0}, "values not 0/1"
        # grow back to 512 and score, like evaluate.py does
        pred = upsample_prob(out, 512) >= 0.5
        s = all_scores(pred, gt)
        print("  %-7s %3dpx  ok  lines %4.1f%%  dice %.3f  precision %.3f  recall %.3f  tol_f1 %.3f"
              % (res, size, 100 * out.mean(), s["dice"], s["precision"], s["recall"], s["tol_f1"]))
        tiles.append(tile(1 - pred.astype(np.float32), res))  # black lines on white
    tiles.append(tile(1 - gt.astype(np.float32), "true mask"))

    out_path = os.path.join(repo, "outputs", "canny_check.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, np.hstack(tiles))
    print("all checks passed; preview saved to", out_path)
