# Role 2: Classical baseline (Canny) + crop sampler

Canny is the "no learning" baseline that the U-Net is compared against. It
plugs into Role 1's `evaluate.py` like any other method. The crop sampler
gives Role 3 augmented training pairs for the U-Net.

## Files

| File | What it does |
| --- | --- |
| `methods/canny.py` | `predict(image, resolution)`: color -> gray -> blur -> Canny -> optional widening. Tuned settings live in `BEST_PARAMS`. `IS_BINARY = True`. |
| `tune_canny.py` | Grid search over Canny settings on 300 validation images. Prints the best settings per resolution. |
| `crop_sampler.py` | `CropDataset` (PyTorch) and `sample_crops` (NumPy): random crops + augmentation for U-Net training. |
| `results.csv` | The 3 Canny rows from the test split (U-Net rows get added here too). |

## Canny

`predict()` follows Role 1's contract: the image comes in already shrunk, and
it returns a 0/1 map of the same size (no resizing inside).

1. **Gray**: Canny only looks at brightness.
2. **Blur** (`sigma`): smooths tiny noise. 0 = no blur.
3. **Canny** (`low`, `high`): finds 1-pixel-wide edges. Edges stronger than `high` are kept; edges between `low` and `high` are kept only if they touch a strong one.
4. **Widen** (`dilate`): grows each line by N pixels (ellipse kernel, same as `metrics.py`).

Tuned settings (best mean Dice on 300 validation images, 152 combinations
tried per resolution):

| Resolution | sigma | low | high | dilate | val Dice |
| --- | --- | --- | --- | --- | --- |
| full (512) | 0 | 400 | 500 | 1 | 0.475 |
| half (256) | 0 | 300 | 400 | 0 | 0.321 |
| quarter (128) | 0 | 250 | 300 | 0 | 0.255 |

Thresholds above 255 are fine: they are compared against edge strength
(Sobel gradient), not pixel brightness.

```bash
python evaluate.py --method canny --resolution all   # the final test run
python tune_canny.py                                 # re-tune (about 2.5 min)
python methods/canny.py                              # quick self-check on one image
```

`tune_canny.py` never edits `canny.py`. It prints a `BEST_PARAMS` block that
you paste in by hand. It uses only `splits["val"]` and writes every
combination to `outputs/tune_canny/grid_<resolution>.csv`.

## Results (test split, 3,545 images)

| Resolution | Dice | IoU | Precision | Recall | Tolerant F1 | ms / image |
| --- | --- | --- | --- | --- | --- | --- |
| full | 0.479 | 0.317 | 0.377 | 0.703 | 0.751 | 0.67 |
| half | 0.324 | 0.196 | 0.299 | 0.376 | 0.635 | 0.26 |
| quarter | 0.257 | 0.148 | 0.197 | 0.397 | 0.483 | 0.16 |

Test Dice is almost the same as validation Dice (0.479 vs 0.475 at full),
so the tuned settings were not overfit to the validation images.

## What we found

* **Double lines.** An artist's line is a dark stripe, so it has two
  brightness jumps, one on each side. Canny draws a thin line at each one.
  That's why plain Dice is low but 1-pixel-tolerant F1 is high. Widening by
  1 pixel merges the pair at full resolution. Widening by 2 makes blobs.
* **Low precision.** Canny also draws shading, texture and every hair strand,
  which an artist would leave out. High thresholds help (precision about
  0.24 -> 0.38 at full) but cannot fix it. Choosing which edges matter is
  the U-Net's job.
* **Resolution.** The best thresholds fall as resolution falls, because
  shrinking weakens edges. No blur wins at every size (clean digital art,
  and shrinking already blurs). Lower resolution scores worse, especially
  tolerant F1, because a 1-pixel line at 128 px becomes a smear when grown
  back to 512.
* **Pale images.** Very light, low-contrast drawings (e.g. `train/1004004`)
  lose most of their lines with one fixed set of thresholds. A learned model
  can adapt per image.
* **Speed.** Under 1 ms per image on a laptop CPU.

## For Role 3: crop sampler

```python
from torch.utils.data import DataLoader
from data import load_splits
from crop_sampler import CropDataset

splits = load_splits()
train_ds = CropDataset(splits["train"], "full")                 # random crops, new every epoch
val_ds = CropDataset(splits["val"], "full", augment=False)      # whole shrunk image, no randomness
train_dl = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=2)

for image, mask in train_dl:   # image: B x 3 x c x c, mask: B x 1 x c x c (both float32, 0-1)
    ...
```

* **Crop size `c`:** full 256, half 256 (whole image), quarter 128 (whole
  image). Change it with `crop_size=...`.
* **Augmentation** (training only):
  * Random crop and a 50% left-right flip, applied to the image AND the mask
    together so they stay lined up.
  * Brightness / contrast / saturation jitter of up to ±20%, applied to the
    image only.
  * No vertical flips or rotation: anime characters are drawn upright.
* **Shrinking** uses Role 1's `downsample_image` / `downsample_mask`, so
  training data is shrunk exactly like the evaluation data.
* `sample_crops(ids, resolution)` returns the same thing as NumPy arrays
  (N x c x c x 3 float32, N x c x c uint8). Only use it for small batches.
* The mask is a float because BCE loss needs floats. Lines are about 5% of
  pixels in full-resolution crops.
* Run `git pull` before your test runs, so your rows get added to the shared
  `results.csv` instead of creating a second copy.

## For Role 4

* **Demo:** `canny.predict(color_np, resolution="full")` works on a 512 x 512
  float RGB image in [0, 1] (what `demo.py` already makes). It returns
  1 = line, so lines show white on black. Use `1 - mask` for black lines on
  white.
* **Figures:** after `evaluate.py --method canny --resolution all`, the
  predicted masks for the first 50 test images are in `outputs/canny/<res>/`.
* **Report table:** the 3 Canny rows are already in `results.csv`.
