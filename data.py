"""Role 1: data loading for the Anime Sketch Colorization Pair dataset.

Each Kaggle file is ONE 512 x 1024 RGB image: the color picture and its
sketch side by side. This module cuts it in half, turns the sketch into a
0/1 line mask, and gives everyone the same resizing rules.

Image IDs look like "train/10023" or "val/10023": folder name + file name
without extension. The folder is part of the ID because the Kaggle train and
val folders could reuse the same file names.

Quick use (in Colab, after mounting Drive):
    from data import load_pair, load_splits
    splits = load_splits()                 # reads splits.json
    color, mask = load_pair(splits["train"][0])
    # color: 512 x 512 x 3 float32 in [0, 1]
    # mask:  512 x 512 uint8, 1 = line, 0 = background
"""

import json
import os

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Settings shared by the whole team
# ---------------------------------------------------------------------------

# Where the dataset lives. Override with:  export ANIME_DATA_ROOT=/some/path
DEFAULT_DATA_ROOT = "/content/drive/MyDrive/ECS174_LineExtractionProject/data/anime"

# Where splits.json lives (in the repo, so everyone uses the same split).
REPO_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SPLITS_PATH = os.path.join(REPO_DIR, "splits.json")

# The three resolutions of our experiment (side length in pixels).
RESOLUTIONS = {"full": 512, "half": 256, "quarter": 128}

# A sketch pixel darker than this (on a 0-1 gray scale) counts as a line.
# Check with check_data.py and change it here if the masks look wrong.
LINE_THRESHOLD = 0.5

# Which half of each file is the sketch: "auto", "left" or "right".
# "auto" picks the half with less color (sketches are black and white).
SKETCH_SIDE = "auto"

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")


# ---------------------------------------------------------------------------
# Finding files
# ---------------------------------------------------------------------------

def get_data_root(data_root=None):
    """Return the folder that contains the Kaggle 'train' and 'val' folders.

    Kaggle zips sometimes add an extra 'data/' level, so we look one and two
    levels down as well.
    """
    root = data_root or os.environ.get("ANIME_DATA_ROOT", DEFAULT_DATA_ROOT)
    candidates = [root, os.path.join(root, "data"),
                  os.path.join(root, "data", "data")]
    for c in candidates:
        if os.path.isdir(os.path.join(c, "train")) and os.path.isdir(os.path.join(c, "val")):
            return c
    raise FileNotFoundError(
        "Could not find 'train' and 'val' folders under %s. "
        "Set ANIME_DATA_ROOT to the folder that contains them." % root
    )


def list_ids(folder, data_root=None):
    """All image IDs in one Kaggle folder ('train' or 'val'), sorted."""
    root = get_data_root(data_root)
    names = sorted(
        f for f in os.listdir(os.path.join(root, folder))
        if f.lower().endswith(IMAGE_EXTENSIONS)
    )
    if not names:
        raise FileNotFoundError(
            "The folder %s has no .png/.jpg images. Is the dataset downloaded "
            "and unzipped there?" % os.path.join(root, folder))
    return [folder + "/" + os.path.splitext(f)[0] for f in names]


def id_to_path(image_id, data_root=None):
    """'train/10023' -> full path of that file (whatever its extension is)."""
    root = get_data_root(data_root)
    base = os.path.join(root, image_id)
    for ext in IMAGE_EXTENSIONS:
        if os.path.exists(base + ext):
            return base + ext
    raise FileNotFoundError("No image file for ID %s (looked for %s.png/.jpg)" % (image_id, base))


# ---------------------------------------------------------------------------
# Loading one pair
# ---------------------------------------------------------------------------

def _read_rgb(path):
    """Read an image file as RGB float32 in [0, 1]."""
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:  # cv2 returns None instead of raising on a bad file
        raise IOError("Could not read image: %s" % path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb.astype(np.float32) / 255.0


def _colorfulness(img):
    """Average saturation: near 0 for a black-and-white sketch."""
    return float(img.max(axis=2).mean() - img.min(axis=2).mean())


def split_halves(full_rgb, sketch_side=SKETCH_SIDE):
    """Cut a side-by-side image into (color, sketch) halves."""
    h, w, _ = full_rgb.shape
    left, right = full_rgb[:, : w // 2], full_rgb[:, w // 2:]
    if sketch_side == "left":
        return right, left
    if sketch_side == "right":
        return left, right
    # auto: the sketch is the less colorful half
    if _colorfulness(left) >= _colorfulness(right):
        return left, right
    return right, left


def sketch_to_mask(sketch_rgb, threshold=LINE_THRESHOLD):
    """Sketch (dark lines on white) -> 0/1 mask where 1 = line."""
    gray = sketch_rgb.mean(axis=2)
    return (gray < threshold).astype(np.uint8)


def load_pair(image_id, data_root=None, line_threshold=LINE_THRESHOLD,
              sketch_side=SKETCH_SIDE):
    """Load one example.

    Returns:
        color: H x W x 3 float32 in [0, 1]   (the model input)
        mask:  H x W uint8, 1 = line          (the ground truth)
    """
    full = _read_rgb(id_to_path(image_id, data_root))
    color, sketch = split_halves(full, sketch_side)
    return np.ascontiguousarray(color), sketch_to_mask(sketch, line_threshold)


# ---------------------------------------------------------------------------
# Resizing rules (use these, do not write your own)
# ---------------------------------------------------------------------------

def downsample_image(img, size):
    """Shrink a color image to size x size WITH anti-aliasing.

    cv2.INTER_AREA averages all the pixels that fall into each new pixel,
    which works like blurring before subsampling (Lecture 3, slides 21-22).
    """
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


def downsample_mask(mask, size):
    """Shrink a 0/1 mask with nearest-neighbor, so it stays exactly 0/1."""
    return cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST)


def upsample_prob(prob, size):
    """Grow a line-probability map back to size x size (bilinear).

    Probabilities are continuous, so smooth interpolation is fine here.
    We threshold AFTER upsampling, so the final mask is still 0/1.
    """
    if prob.shape[0] == size and prob.shape[1] == size:
        return prob
    return cv2.resize(prob.astype(np.float32), (size, size),
                      interpolation=cv2.INTER_LINEAR)


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------

def load_splits(path=DEFAULT_SPLITS_PATH):
    """Read splits.json -> {"train": [...], "val": [...], "test": [...]}."""
    with open(path) as f:
        return json.load(f)
