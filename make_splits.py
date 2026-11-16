"""Role 1: make splits.json (run ONCE, then commit splits.json to GitHub).

Rules:
  * test  = every image in the Kaggle 'val' folder (3,545 images)
  * train / val = the Kaggle 'train' folder (14,224 images), shuffled with a
    fixed seed and cut 90% / 10%
  * We split by WHOLE IMAGE before anyone cuts crops, so no crop of a test
    image can ever end up in training (no data leakage).

Usage (Colab, after mounting Drive):
    python make_splits.py
    python make_splits.py --data-root /content/anime   # if you copied the data
"""

import argparse
import json
import random

from data import DEFAULT_SPLITS_PATH, list_ids

SEED = 174          # fixed, so everyone gets the same split
VAL_FRACTION = 0.10


def make_splits(data_root=None, seed=SEED, val_fraction=VAL_FRACTION):
    train_folder = list_ids("train", data_root)
    test = list_ids("val", data_root)

    rng = random.Random(seed)
    shuffled = train_folder[:]          # copy, keep the sorted list intact
    rng.shuffle(shuffled)

    n_val = round(len(shuffled) * val_fraction)
    val = sorted(shuffled[:n_val])
    train = sorted(shuffled[n_val:])

    # Safety checks: no image in two splits, nothing lost.
    assert not (set(train) & set(val)), "train and val overlap"
    assert not (set(train) & set(test)), "train and test overlap"
    assert not (set(val) & set(test)), "val and test overlap"
    assert len(train) + len(val) == len(train_folder)

    return {"seed": seed, "train": train, "val": val, "test": test}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--out", default=DEFAULT_SPLITS_PATH)
    args = parser.parse_args()

    splits = make_splits(args.data_root)
    with open(args.out, "w") as f:
        json.dump(splits, f, indent=1)
    print("Wrote %s: train=%d, val=%d, test=%d" % (
        args.out, len(splits["train"]), len(splits["val"]), len(splits["test"])))


if __name__ == "__main__":
    main()
