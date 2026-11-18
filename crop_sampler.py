"""Role 2: crop sampler + paired augmentation for Role 3's U-Net.

Gives the U-Net (color image, line mask) training pairs:
  load (512 x 512) -> shrink to the resolution -> random crop -> random
  left-right flip -> random color change (image only)

Every call gives a slightly different version of the same picture, so the
network sees more variety and memorizes less (data augmentation, Lecture 7
slide 77).

THE GOLDEN RULE: anything that MOVES pixels (crop, flip) is done to the image
AND the mask in exactly the same way, or the mask no longer lines up with the
picture. Anything that only changes COLORS is done to the image only (making
a picture darker does not move its lines).

Two ways to use it (same augmentation inside):
  * sample_crops(ids, resolution)  -> NumPy arrays (small batches, checking)
  * CropDataset(ids, resolution)   -> PyTorch Dataset (training with a
                                      DataLoader; loads one image at a time)

Use splits["train"] for training, splits["val"] with augment=False for
validation. Never splits["test"].
"""

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from data import RESOLUTIONS, downsample_image, downsample_mask, load_pair

# Crop size per resolution (from the project plan): 256 x 256 crops at full;
# at half (256) and quarter (128) the whole image is used, since it is
# already that small.
CROP_SIZES = {"full": 256, "half": 256, "quarter": 128}

# How strong the random color change can be: each factor is picked
# uniformly from [1 - JITTER, 1 + JITTER], i.e. up to +/-20%.
JITTER = 0.2


def random_crop(color, mask, crop_size, rng):
    """Cut the SAME random crop_size x crop_size square out of both.

    color: H x W x 3, mask: H x W. If the image is already crop_size or
    smaller, both are returned unchanged (nothing to crop).
    """
    h, w = mask.shape
    if crop_size >= h and crop_size >= w:
        return color, mask
    top = rng.integers(0, h - crop_size + 1)     # +1: the last valid corner is h - crop_size
    left = rng.integers(0, w - crop_size + 1)
    return (color[top:top + crop_size, left:left + crop_size],
            mask[top:top + crop_size, left:left + crop_size])


def random_flip(color, mask, rng):
    """With 50% chance, mirror BOTH left-right.

    A mirrored character is still a valid drawing with valid line art.
    (No upside-down flips: characters are never drawn upside down.)
    """
    if rng.random() < 0.5:
        # [:, ::-1] reverses the columns. ascontiguousarray makes a real copy,
        # because PyTorch's torch.from_numpy cannot take "reversed view" arrays.
        color = np.ascontiguousarray(color[:, ::-1])
        mask = np.ascontiguousarray(mask[:, ::-1])
    return color, mask


def color_jitter(color, rng, strength=JITTER):
    """Random brightness, contrast and saturation change. IMAGE ONLY.

    color: H x W x 3 float32 RGB in [0, 1]. Each factor is picked from
    [1 - strength, 1 + strength]; 1 means "no change".
      brightness: multiply every pixel          (darker / lighter)
      contrast:   push pixels away from / toward the image's mean
      saturation: push colors away from / toward their own gray value
    The result is clipped back to [0, 1].
    """
    def factor():
        return rng.uniform(1 - strength, 1 + strength)

    out = color * factor()                                    # brightness
    mean = out.mean()
    out = mean + (out - mean) * factor()                      # contrast
    gray = cv2.cvtColor(out.astype(np.float32), cv2.COLOR_RGB2GRAY)[..., None]
    out = gray + (out - gray) * factor()                      # saturation
    return np.clip(out, 0, 1).astype(np.float32)


def make_pair(image_id, resolution, crop_size=None, rng=None, augment=True):
    """Load one example and prepare it for the U-Net.

    1. load the 512 x 512 color image and mask (Role 1's load_pair)
    2. shrink both to the resolution with Role 1's rules
       (image: anti-aliased INTER_AREA, mask: nearest so it stays 0/1)
    3. if augment: random crop -> random flip -> color jitter
       if not augment: the whole shrunk image, unchanged (for validation)

    Returns color (c x c x 3 float32, 0-1) and mask (c x c uint8, 0/1).
    """
    if crop_size is None:
        crop_size = CROP_SIZES[resolution]
    if rng is None:
        rng = np.random.default_rng()

    color, mask = load_pair(image_id)
    size = RESOLUTIONS[resolution]
    color = downsample_image(color, size)       # shrink BEFORE cropping, so a crop
    mask = downsample_mask(mask, size)          # covers the same area as at test time

    if augment:
        color, mask = random_crop(color, mask, crop_size, rng)   # moves pixels: both
        color, mask = random_flip(color, mask, rng)              # moves pixels: both
        color = color_jitter(color, rng)                         # colors only: image
    return color, mask


def sample_crops(ids, resolution, crop_size=None, rng=None, augment=True):
    """make_pair for every ID, stacked into arrays (the plan's interface).

    Returns:
        colors: N x c x c x 3 float32 in [0, 1]
        masks:  N x c x c uint8, 1 = line
    Holds everything in memory, so use it for small batches. For training on
    all 12,802 images use CropDataset.
    """
    if rng is None:
        rng = np.random.default_rng()
    pairs = [make_pair(i, resolution, crop_size, rng, augment) for i in ids]
    colors = np.stack([c for c, _ in pairs]).astype(np.float32)
    masks = np.stack([m for _, m in pairs]).astype(np.uint8)
    return colors, masks


class CropDataset(Dataset):
    """PyTorch Dataset version for training with a DataLoader.

    dataset[i] loads image ids[i] fresh from disk and returns
        image: 3 x c x c float32 tensor in [0, 1]  (PyTorch wants channels first)
        mask:  1 x c x c float32 tensor, 0/1       (float, because BCE loss needs floats)

    seed=None (default, use for training): a NEW random crop/flip/jitter every
        time an image is loaded, so each epoch sees different versions. It also
        stays random with num_workers > 0 (each call makes its own generator).
    seed=<int>: the same augmentation for each index every time (for checking
        and debugging only; training with it would remove the variety).
    """

    def __init__(self, ids, resolution, crop_size=None, augment=True, seed=None):
        self.ids = list(ids)
        self.resolution = resolution
        self.crop_size = crop_size
        self.augment = augment
        self.seed = seed

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        if self.seed is None:
            rng = np.random.default_rng()
        else:
            rng = np.random.default_rng([self.seed, index])
        color, mask = make_pair(self.ids[index], self.resolution, self.crop_size, rng, self.augment)
        image = torch.from_numpy(np.ascontiguousarray(color.transpose(2, 0, 1)))   # H x W x 3 -> 3 x H x W
        mask = torch.from_numpy(mask[None].astype(np.float32))                    # H x W -> 1 x H x W
        return image, mask


if __name__ == "__main__":
    # Self-check. Run from team_repo/:
    #   ANIME_DATA_ROOT=../data ../.venv/bin/python crop_sampler.py
    import os
    import random

    from torch.utils.data import DataLoader

    from data import load_splits

    # 1. Golden rule test. Fake image whose mask is COMPUTED from the image
    #    (mask = red channel > 0.5). If crop and flip treat both the same,
    #    that is still true afterwards, for every random choice.
    fake_rng = np.random.default_rng(0)
    fake = fake_rng.random((512, 512, 3)).astype(np.float32)
    fake_mask = (fake[..., 0] > 0.5).astype(np.uint8)
    for s in range(200):
        r = np.random.default_rng(s)
        c, m = random_crop(fake, fake_mask, 256, r)
        c, m = random_flip(c, m, r)
        assert np.array_equal(m, (c[..., 0] > 0.5).astype(np.uint8)), "image and mask out of line!"
    print("golden rule: crop + flip keep image and mask aligned (200 random tries)  ok")

    # 2. Shapes and types on real training images, all 3 resolutions.
    ids = random.Random(174).sample(load_splits()["train"], 4)   # 4 fixed training images
    for res in RESOLUTIONS:
        c = CROP_SIZES[res]
        colors, masks = sample_crops(ids, res, rng=np.random.default_rng(0))
        assert colors.shape == (4, c, c, 3) and colors.dtype == np.float32
        assert masks.shape == (4, c, c) and masks.dtype == np.uint8
        assert colors.min() >= 0 and colors.max() <= 1 and set(np.unique(masks)) <= {0, 1}
        full = RESOLUTIONS[res]
        vc, vm = sample_crops(ids[:2], res, augment=False)
        assert vc.shape == (2, full, full, 3) and vm.shape == (2, full, full)
        print("%-8s sample_crops %s / %s, augment=False -> whole %d px image  ok"
              % (res, colors.shape, masks.shape, full))

    # 3. The PyTorch Dataset through a DataLoader.
    loader = DataLoader(CropDataset(ids, "full"), batch_size=4, shuffle=True)
    images, batch_masks = next(iter(loader))
    assert images.shape == (4, 3, 256, 256) and images.dtype == torch.float32
    assert batch_masks.shape == (4, 1, 256, 256) and batch_masks.dtype == torch.float32
    seeded = CropDataset(ids, "full", seed=1)
    assert torch.equal(seeded[0][0], seeded[0][0]), "seed should repeat the same crop"
    print("CropDataset + DataLoader batch: images %s, masks %s  ok"
          % (tuple(images.shape), tuple(batch_masks.shape)))

    # 4. Preview: each color crop with its mask drawn on top in red. If the red
    #    lines sit on the drawn lines, the pairs line up.
    tiles = []
    for res in RESOLUTIONS:
        colors, masks = sample_crops(ids, res, rng=np.random.default_rng(1))
        for c, m in zip(colors, masks):
            over = c.copy()
            over[m == 1] = [1, 0, 0]
            for img in (c, over):
                t = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
                t = cv2.resize(t, (192, 192), interpolation=cv2.INTER_NEAREST)
                cv2.putText(t, res, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                tiles.append(t)
    rows = [np.hstack(tiles[i:i + 8]) for i in range(0, len(tiles), 8)]
    os.makedirs("outputs", exist_ok=True)
    path = os.path.join("outputs", "crop_check.png")
    cv2.imwrite(path, np.vstack(rows))
    print("all checks passed; preview saved to", os.path.abspath(path))
