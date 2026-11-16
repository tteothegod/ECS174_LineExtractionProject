"""Role 1: score any line-extraction method the same, fair way.

How it works for each image:
  1. load the 512 x 512 color image and its true 0/1 line mask
  2. shrink the color image to the chosen resolution (anti-aliased)
  3. call the method's predict() -> line probability map at that size
  4. grow the map back to 512 x 512, then threshold it into a 0/1 mask
  5. score it against the ORIGINAL 512 x 512 true mask
The true mask is never resized, so all 6 conditions are scored the same way.

The threshold (probability above which a pixel counts as a line) is chosen
on VALIDATION images only, then used unchanged on the test images.

A method is a file methods/<name>.py with:
    def predict(image, resolution):
        # image: H x W x 3 float32 in [0, 1], already shrunk to the resolution
        # resolution: "full", "half" or "quarter" (so a U-Net can pick its
        #             checkpoint for that size)
        # return: H x W float array in [0, 1], same H and W as image
    IS_BINARY = True   # optional: set if predict() already returns 0/1
                       # (e.g. Canny); then no threshold is tuned

Usage (Colab):
    python evaluate.py --method canny --resolution full
    python evaluate.py --method unet --resolution all
    python evaluate.py --method canny --resolution half --limit 100   # quick try

Writes:
    results.csv                                   one row per run (the table)
    outputs/<method>/<resolution>/per_image_<split>.csv   one row per image
    outputs/<method>/<resolution>/*.png           predicted masks for the first
                                                  --save-n test images (figures)
"""

import argparse
import csv
import importlib
import os
import time

import cv2
import numpy as np

from data import (RESOLUTIONS, downsample_image, load_pair, load_splits,
                  upsample_prob)
from metrics import all_scores, dice

THRESHOLD_GRID = [round(0.05 * k, 2) for k in range(1, 20)]   # 0.05, 0.10, ..., 0.95
SCORE_NAMES = ["dice", "iou", "precision", "recall",
               "tol_precision", "tol_recall", "tol_f1"]


def load_method(name):
    """Import methods/<name>.py (or <name>.py at the repo top level)."""
    for module_name in ("methods." + name, name):
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError as e:
            if e.name in (module_name, "methods"):
                continue
            raise  # the method file exists but something inside it is missing
        if not hasattr(module, "predict"):
            raise AttributeError("%s.py has no predict(image, resolution)" % name)
        return module
    raise ModuleNotFoundError("No methods/%s.py found" % name)


def run_predict(method, color, resolution):
    """Shrink -> predict -> grow back. Returns (prob at 512, seconds)."""
    size = RESOLUTIONS[resolution]
    small = downsample_image(color, size)
    start = time.perf_counter()
    prob = method.predict(small, resolution)
    seconds = time.perf_counter() - start

    prob = np.asarray(prob, dtype=np.float32)
    if prob.ndim == 3 and prob.shape[-1] == 1:
        prob = prob[..., 0]
    if prob.shape != small.shape[:2]:
        raise ValueError("predict() returned shape %s, expected %s"
                         % (prob.shape, small.shape[:2]))
    if prob.min() < 0 or prob.max() > 1:
        raise ValueError("predict() must return values in [0, 1], got [%.3f, %.3f]"
                         % (prob.min(), prob.max()))
    return upsample_prob(prob, color.shape[0]), seconds


def tune_threshold(method, resolution, val_ids, data_root=None):
    """Pick the threshold with the best average Dice on validation images."""
    if getattr(method, "IS_BINARY", False):
        return 0.5
    totals = np.zeros(len(THRESHOLD_GRID))
    for i, image_id in enumerate(val_ids):
        color, gt = load_pair(image_id, data_root)
        prob, _ = run_predict(method, color, resolution)
        for k, t in enumerate(THRESHOLD_GRID):
            totals[k] += dice(prob >= t, gt)
        if (i + 1) % 100 == 0:
            print("  tuning: %d / %d validation images" % (i + 1, len(val_ids)))
    best = int(np.argmax(totals))
    print("  best threshold on validation: %.2f (mean Dice %.4f)"
          % (THRESHOLD_GRID[best], totals[best] / len(val_ids)))
    return THRESHOLD_GRID[best]


def safe_name(image_id):
    return image_id.replace("/", "_")


def evaluate(method_name, resolution, split="test", threshold=None,
             limit=None, tune_limit=300, save_n=50, tolerance=1,
             data_root=None, results_csv="results.csv", out_dir="outputs"):
    method = load_method(method_name)
    splits = load_splits()
    ids = splits[split][:limit] if limit else splits[split]

    print("== %s @ %s (%dpx) on %d %s images" % (
        method_name, resolution, RESOLUTIONS[resolution], len(ids), split))
    if threshold is None:
        threshold = tune_threshold(method, resolution,
                                   splits["val"][:tune_limit], data_root)

    run_dir = os.path.join(out_dir, method_name, resolution)
    os.makedirs(run_dir, exist_ok=True)

    rows, seconds = [], []
    for i, image_id in enumerate(ids):
        color, gt = load_pair(image_id, data_root)
        prob, sec = run_predict(method, color, resolution)
        pred = (prob >= threshold).astype(np.uint8)
        scores = all_scores(pred, gt, tolerance)
        rows.append(dict(id=image_id, **scores))
        seconds.append(sec)
        if i < save_n:
            cv2.imwrite(os.path.join(run_dir, safe_name(image_id) + ".png"),
                        (1 - pred) * 255)  # black lines on white, like a sketch
        if (i + 1) % 200 == 0:
            print("  %d / %d done" % (i + 1, len(ids)))

    with open(os.path.join(run_dir, "per_image_%s.csv" % split), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id"] + SCORE_NAMES)
        writer.writeheader()
        writer.writerows(rows)

    summary = {"method": method_name, "resolution": resolution, "split": split,
               "n_images": len(rows), "threshold": threshold, "tolerance_px": tolerance}
    for name in SCORE_NAMES:
        summary[name] = round(float(np.mean([r[name] for r in rows])), 4)
    summary["ms_per_image"] = round(1000 * float(np.mean(seconds)), 2)

    new_file = not os.path.exists(results_csv)
    with open(results_csv, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary))
        if new_file:
            writer.writeheader()
        writer.writerow(summary)

    print("  " + ", ".join("%s=%s" % (k, summary[k]) for k in
                           ["dice", "iou", "precision", "recall", "tol_f1", "ms_per_image"]))
    return summary


def main():
    parser = argparse.ArgumentParser(description="Score a method on our split.")
    parser.add_argument("--method", required=True, help="file name in methods/, e.g. canny")
    parser.add_argument("--resolution", default="full",
                        choices=list(RESOLUTIONS) + ["all"])
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--threshold", type=float, default=None,
                        help="fixed threshold; default = tune on validation")
    parser.add_argument("--limit", type=int, default=None, help="only the first N images")
    parser.add_argument("--tune-limit", type=int, default=300,
                        help="validation images used to pick the threshold")
    parser.add_argument("--save-n", type=int, default=50)
    parser.add_argument("--tolerance", type=int, default=1, help="pixels, for tol_* scores")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--results", default="results.csv")
    args = parser.parse_args()

    resolutions = list(RESOLUTIONS) if args.resolution == "all" else [args.resolution]
    for res in resolutions:
        evaluate(args.method, res, args.split, args.threshold, args.limit,
                 args.tune_limit, args.save_n, args.tolerance, args.data_root,
                 args.results)


if __name__ == "__main__":
    main()
