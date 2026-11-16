"""Role 1: look at the data before trusting it.

Saves outputs/check_data.png with a few examples: color half | sketch half |
our 0/1 line mask. Open it and check:
  * the color and sketch halves are not swapped
  * the mask has the lines (white) and not the background
  * thin lines are not lost (if they are, raise LINE_THRESHOLD in data.py)

Also prints how many pixels are lines, which tells Role 3 how unbalanced
the classes are.

Usage:  python check_data.py --n 6
"""

import argparse
import os

import cv2
import numpy as np

from data import _read_rgb, id_to_path, list_ids, split_halves, sketch_to_mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--n", type=int, default=6)
    parser.add_argument("--out", default="outputs/check_data.png")
    args = parser.parse_args()

    ids = list_ids("train", args.data_root)
    print("Kaggle train folder: %d images" % len(ids))
    print("Kaggle val folder:   %d images" % len(list_ids("val", args.data_root)))

    rows, fractions = [], []
    step = max(1, len(ids) // args.n)
    for image_id in ids[::step][: args.n]:
        full = _read_rgb(id_to_path(image_id, args.data_root))
        color, sketch = split_halves(full)
        mask = sketch_to_mask(sketch)
        fractions.append(mask.mean())
        mask_rgb = np.repeat(mask[:, :, None].astype(np.float32), 3, axis=2)
        rows.append(np.concatenate([color, sketch, mask_rgb], axis=1))
        print("%s: file %dx%d, line pixels = %.1f%%" % (
            image_id, full.shape[1], full.shape[0], 100 * mask.mean()))

    print("Average line pixels: %.1f%% of each image" % (100 * np.mean(fractions)))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    grid = (np.concatenate(rows, axis=0) * 255).astype(np.uint8)
    cv2.imwrite(args.out, cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
    print("Saved", args.out, "(columns: color | sketch | line mask)")


if __name__ == "__main__":
    main()
