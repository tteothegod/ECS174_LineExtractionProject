"""Tests for Role 1 code. Run from the repo folder:  python -m pytest tests -q
(or without pytest:  python tests/test_role1.py)

The pipeline test builds a tiny fake dataset with the same layout as Kaggle
(side-by-side color | sketch files in train/ and val/), so it runs anywhere,
without Google Drive.
"""

import json
import os
import shutil
import sys
import tempfile

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import data
import metrics


# --------------------------------------------------------------------------
# Metrics on toy masks
# --------------------------------------------------------------------------

def test_perfect_match():
    m = np.zeros((10, 10), np.uint8)
    m[2:8, 5] = 1
    s = metrics.all_scores(m, m)
    assert s["dice"] == 1.0 and s["iou"] == 1.0
    assert s["precision"] == 1.0 and s["recall"] == 1.0


def test_no_overlap():
    gt = np.zeros((10, 10), np.uint8)
    gt[:, 1] = 1
    pred = np.zeros((10, 10), np.uint8)
    pred[:, 8] = 1
    s = metrics.all_scores(pred, gt)
    assert s["dice"] == 0.0 and s["iou"] == 0.0
    assert s["precision"] == 0.0 and s["recall"] == 0.0


def test_proposal_example():
    # 100 true line pixels, 80 predicted, 60 of them correct
    gt = np.zeros(200, np.uint8)
    gt[:100] = 1
    pred = np.zeros(200, np.uint8)
    pred[40:120] = 1          # overlaps gt on 40..99 = 60 pixels
    assert abs(metrics.precision(pred, gt) - 0.75) < 1e-9
    assert abs(metrics.recall(pred, gt) - 0.60) < 1e-9
    assert abs(metrics.dice(pred, gt) - 120 / 180) < 1e-9
    assert abs(metrics.iou(pred, gt) - 0.50) < 1e-9


def test_both_empty_is_perfect():
    z = np.zeros((5, 5), np.uint8)
    assert metrics.dice(z, z) == 1.0


def test_tolerance_forgives_one_pixel_shift():
    gt = np.zeros((20, 20), np.uint8)
    gt[:, 10] = 1
    pred = np.zeros((20, 20), np.uint8)
    pred[:, 11] = 1           # same line, one pixel to the right
    assert metrics.dice(pred, gt) == 0.0                     # strict: all wrong
    assert metrics.tolerant_scores(pred, gt, 1)["tol_f1"] == 1.0   # tolerant: right
    assert metrics.tolerant_scores(pred, gt, 0)["tol_f1"] == 0.0


# --------------------------------------------------------------------------
# Data loading and resizing
# --------------------------------------------------------------------------

def _fake_pair(seed, size=64, sketch_on_right=True):
    """A colorful image and a matching black-on-white sketch."""
    rng = np.random.default_rng(seed)
    color = np.zeros((size, size, 3), np.uint8)
    color[:] = rng.integers(60, 255, size=3)                 # colored background
    sketch = np.full((size, size, 3), 255, np.uint8)
    x = int(rng.integers(5, size - 5))
    cv2.line(color, (x, 0), (x, size - 1), (20, 20, 20), 2)
    cv2.line(sketch, (x, 0), (x, size - 1), (0, 0, 0), 2)
    halves = [color, sketch] if sketch_on_right else [sketch, color]
    return np.concatenate(halves, axis=1)


def _make_fake_dataset(root, n_train=30, n_val=8, sketch_on_right=True):
    for folder, n, offset in (("train", n_train, 0), ("val", n_val, 1000)):
        os.makedirs(os.path.join(root, folder))
        for i in range(n):
            img = _fake_pair(offset + i, sketch_on_right=sketch_on_right)
            cv2.imwrite(os.path.join(root, folder, "%d.png" % (offset + i)), img)


def test_load_pair_and_sketch_side_detection():
    for sketch_on_right in (True, False):
        root = tempfile.mkdtemp()
        try:
            _make_fake_dataset(root, 3, 1, sketch_on_right)
            color, mask = data.load_pair("train/0", root)
            assert color.shape == (64, 64, 3) and color.dtype == np.float32
            assert mask.shape == (64, 64) and set(np.unique(mask)) <= {0, 1}
            # the colorful half was picked as the input, not the sketch
            assert color.max(axis=2).mean() - color.min(axis=2).mean() > 0.05
            # the mask is the thin line, not the white background
            assert 0 < mask.mean() < 0.2
        finally:
            shutil.rmtree(root)


def test_resizing_rules():
    mask = np.zeros((512, 512), np.uint8)
    mask[::7, :] = 1
    small = data.downsample_mask(mask, 128)
    assert small.shape == (128, 128) and set(np.unique(small)) <= {0, 1}
    img = np.random.rand(512, 512, 3).astype(np.float32)
    assert data.downsample_image(img, 256).shape == (256, 256, 3)
    prob = np.random.rand(128, 128).astype(np.float32)
    assert data.upsample_prob(prob, 512).shape == (512, 512)


# --------------------------------------------------------------------------
# Full pipeline: make_splits -> evaluate
# --------------------------------------------------------------------------

def test_pipeline_end_to_end():
    import evaluate
    import make_splits

    root = tempfile.mkdtemp()
    work = tempfile.mkdtemp()
    old_cwd, old_splits = os.getcwd(), data.DEFAULT_SPLITS_PATH
    try:
        _make_fake_dataset(root)
        splits = make_splits.make_splits(root)
        assert len(splits["test"]) == 8
        assert len(splits["val"]) == 3 and len(splits["train"]) == 27
        # same seed -> same split every time
        assert make_splits.make_splits(root) == splits

        splits_path = os.path.join(work, "splits.json")
        with open(splits_path, "w") as f:
            json.dump(splits, f)
        os.chdir(work)
        evaluate.load_splits = lambda: data.load_splits(splits_path)
        # fake images are 64 px, so evaluate at "sizes" that fit them
        old_res = dict(data.RESOLUTIONS)
        evaluate.RESOLUTIONS.update({"full": 64, "half": 32, "quarter": 16})
        try:
            for res in ("full", "half", "quarter"):
                s = evaluate.evaluate("dark_pixels", res, data_root=root, save_n=2)
                assert s["n_images"] == 8 and 0.0 <= s["dice"] <= 1.0
        finally:
            evaluate.RESOLUTIONS.clear()
            evaluate.RESOLUTIONS.update(old_res)
        with open("results.csv") as f:
            assert len(f.read().strip().splitlines()) == 4   # header + 3 rows
        assert os.path.exists("outputs/dark_pixels/full/per_image_test.csv")
        assert isinstance(s["threshold"], float)
    finally:
        evaluate.load_splits = data.load_splits
        os.chdir(old_cwd)
        data.DEFAULT_SPLITS_PATH = old_splits
        shutil.rmtree(root)
        shutil.rmtree(work)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
