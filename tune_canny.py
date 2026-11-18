"""Role 2: pick the best Canny settings for each resolution (grid search).

For each resolution (full 512, half 256, quarter 128) we try every
combination of sigma / low / high / dilate on VALIDATION images and keep the
one with the highest mean Dice. Test images are never used here, so the final
test score stays an honest estimate.

Each combination is scored exactly the way evaluate.py scores a method:
shrink the color image -> predict() -> grow the map back to 512 ->
threshold at 0.5 -> Dice against the original 512 x 512 true mask.

This script does NOT edit methods/canny.py. It prints a BEST_PARAMS block
that we paste into canny.py by hand.

Usage (from team_repo/):
    ANIME_DATA_ROOT=../data ../.venv/bin/python tune_canny.py
    ANIME_DATA_ROOT=../data ../.venv/bin/python tune_canny.py --resolution quarter --n-images 50

Writes:
    outputs/tune_canny/grid_<resolution>.csv   every combination, best first
"""

import argparse
import csv
import itertools
import os
import random
import time

import numpy as np

from data import RESOLUTIONS, downsample_image, load_pair, load_splits, upsample_prob
from methods.canny import predict
from metrics import dice, precision, recall

# The values to try. The thresholds were raised twice because the winners kept
# landing on the largest value tried (see docs/tune_canny.md). Edge strength
# can go far above 255, so thresholds of 300-600 are valid.
SIGMAS = [0, 0.5, 1, 2]        # blur strength (0 = no blur)
LOWS = [150, 200, 250, 300, 400]   # weak-edge threshold
HIGHS = [250, 300, 400, 500, 600]  # strong-edge threshold
DILATES = [0, 1]               # line widening (2 made blobs in testing)

SEED = 174                     # the team's seed, so the 300 images are always the same
OUT_DIR = os.path.join("outputs", "tune_canny")


def make_grid():
    """Every combination of the 4 settings, as a list of dicts.

    Combinations with low >= high are skipped: Canny needs the weak-edge
    threshold below the strong-edge one. That leaves 152 combinations.
    """
    grid = []
    for sigma, low, high, dilate in itertools.product(SIGMAS, LOWS, HIGHS, DILATES):
        if low < high:
            grid.append({"sigma": sigma, "low": low, "high": high, "dilate": dilate})
    return grid


def pick_val_ids(n):
    """A random but fixed set of n validation IDs.

    splits["val"] is sorted alphabetically, so the first n would not be a
    fair mix. A seeded random sample gives the same n images every run.
    """
    val = load_splits()["val"]
    return random.Random(SEED).sample(val, min(n, len(val)))


def score_image(small, gt, resolution, params):
    """Score ONE shrunk image with ONE setting, the same way evaluate.py does.

    small: color image already shrunk to the resolution
    gt:    the original 512 x 512 true mask (never resized)
    Returns (dice, precision, recall).
    """
    out = predict(small, resolution, **params)        # 0/1 map at the small size
    pred = upsample_prob(out, gt.shape[0]) >= 0.5      # grow to 512, back to 0/1
    return dice(pred, gt), precision(pred, gt), recall(pred, gt)


def save_csv(path, rows):
    """Write one resolution's results, best Dice first."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Grid-search Canny settings on validation images.")
    parser.add_argument("--resolution", default="all", choices=list(RESOLUTIONS) + ["all"])
    parser.add_argument("--n-images", type=int, default=300, help="validation images to use")
    args = parser.parse_args()

    resolutions = list(RESOLUTIONS) if args.resolution == "all" else [args.resolution]
    grid = make_grid()
    ids = pick_val_ids(args.n_images)
    print("trying %d settings x %d resolutions on %d validation images"
          % (len(grid), len(resolutions), len(ids)))

    # Running totals: totals[res][k] = [sum of dice, sum of precision, sum of recall]
    # for setting number k. We load one image at a time and add its scores, so
    # memory stays small (300 full-size images at once would be about 0.9 GB).
    totals = {res: np.zeros((len(grid), 3)) for res in resolutions}

    start = time.time()
    for i, image_id in enumerate(ids):
        color, gt = load_pair(image_id)
        for res in resolutions:
            small = downsample_image(color, RESOLUTIONS[res])   # shrink once, reuse for every setting
            for k, params in enumerate(grid):
                totals[res][k] += score_image(small, gt, res, params)
        if (i + 1) % 25 == 0 or i + 1 == len(ids):
            print("  %d / %d images  (%.0f s)" % (i + 1, len(ids), time.time() - start))

    best = {}
    for res in resolutions:
        means = totals[res] / len(ids)                       # sums -> averages
        rows = [dict(params, dice=round(m[0], 4), precision=round(m[1], 4), recall=round(m[2], 4))
                for params, m in zip(grid, means)]
        rows.sort(key=lambda r: r["dice"], reverse=True)    # best Dice first
        save_csv(os.path.join(OUT_DIR, "grid_%s.csv" % res), rows)
        best[res] = rows[0]

        print("\n%s (%dpx): top 5 by mean Dice" % (res, RESOLUTIONS[res]))
        print("   sigma  low  high  dilate   dice   prec  recall")
        for r in rows[:5]:
            print("   %5s  %3d  %4d  %6d   %.3f  %.3f  %.3f"
                  % (r["sigma"], r["low"], r["high"], r["dilate"], r["dice"], r["precision"], r["recall"]))

    print("\nPaste into methods/canny.py:")
    print("BEST_PARAMS = {")
    for res in resolutions:
        b = best[res]
        print('    %-10s {"sigma": %s, "low": %d, "high": %d, "dilate": %d},   # val Dice %.3f'
              % ('"%s":' % res, b["sigma"], b["low"], b["high"], b["dilate"], b["dice"]))
    print("}")
    print("all results saved in", OUT_DIR)


if __name__ == "__main__":
    main()
