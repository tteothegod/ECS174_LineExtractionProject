# Role 1: Data split + evaluation

Everything that loads data or computes a score lives here. Roles 2 and 3 only
need to write one `predict()` function each; `evaluate.py` does the rest, the
same way for every method, so the comparison is fair.

## Files

| File | What it does |
| --- | --- |
| `data.py` | `load_pair(id)` -> color image (512x512x3, float 0-1) + line mask (512x512, 1 = line). Also the shared resizing rules and `load_splits()`. |
| `make_splits.py` | Makes `splits.json` (run once, then commit it). |
| `check_data.py` | Saves `outputs/check_data.png` (color \| sketch \| mask) so we can eyeball the masks. |
| `metrics.py` | Dice, IoU, precision, recall, plus 1-pixel-tolerant precision / recall / F1. |
| `evaluate.py` | Runs any method on the test set and appends one row to `results.csv`. |
| `methods/dark_pixels.py` | A dummy method ("dark = line") to test the pipeline and to copy from. |
| `tests/test_role1.py` | Toy-mask tests + a full run on a fake dataset. `python tests/test_role1.py` |

## The split

* **test** = all 3,545 images in the Kaggle `val` folder
* **train / val** = the 14,224 images in the Kaggle `train` folder, shuffled with seed 174, 90% / 10% (12,802 / 1,422)
* Split by whole image before any cropping, so no piece of a test image is ever trained on.
* Image IDs look like `train/10023`: folder + file name without extension.

```python
from data import load_pair, load_splits
splits = load_splits()
color, mask = load_pair(splits["train"][0])
```

## For Role 2 (Canny) and Role 3 (U-Net): the one thing you write

Make `methods/canny.py` or `methods/unet.py` with:

```python
def predict(image, resolution):
    # image: H x W x 3 float32 in [0, 1], ALREADY shrunk to the resolution
    #        (512 for "full", 256 for "half", 128 for "quarter")
    # resolution: "full", "half" or "quarter"  (U-Net: load that checkpoint)
    # return: H x W float array in [0, 1], same H and W as image
    ...

IS_BINARY = True   # Canny only: predict() already returns 0/1, skip threshold tuning
```

Do **not** resize inside `predict()` and do not threshold the U-Net output: `evaluate.py` shrinks the input, grows your output back to 512, picks the threshold on validation images, and scores against the original mask. Load the U-Net weights once (at import time or with a cache), not on every call, or the timing will be wrong. If you use the GPU, call `torch.cuda.synchronize()` before returning so the measured time is real.

For training (Role 3) and the crop sampler (Role 2), use `load_pair()` with `splits["train"]` and pick epochs / Canny settings with `splits["val"]`. Never use `splits["test"]` until the final run.

Use the resizing helpers in `data.py` (`downsample_image`, `downsample_mask`) so training data is shrunk exactly like the evaluation data.

## Running the evaluation

```bash
python evaluate.py --method canny --resolution full --limit 100   # quick try
python evaluate.py --method canny --resolution all                # all 3 sizes
python evaluate.py --method unet  --resolution all
```

Output:

* `results.csv`: one row per run: method, resolution, threshold, Dice, IoU, precision, recall, tolerant precision / recall / F1, ms per image. This becomes the 6-row table in the report.
* `outputs/<method>/<resolution>/per_image_test.csv`: scores per image (for error bars).
* `outputs/<method>/<resolution>/*.png`: predicted masks for the first 50 test images (Role 4: figures).

## For Role 4

* Figures: the saved PNGs are black lines on white; the matching inputs come from `load_pair(id)`. File names are the IDs with `/` replaced by `_`.
* Timing: `ms_per_image` in `results.csv` times only `predict()`; compare methods on the same Colab GPU.

## Speed tip (Colab)

Reading 3,545 images one by one from Google Drive is slow. Copy once to the Colab disk, then point the code at it:

```bash
!cp -r /content/drive/MyDrive/ECS174_LineExtractionProject/data/anime /content/anime
%env ANIME_DATA_ROOT=/content/anime
```
