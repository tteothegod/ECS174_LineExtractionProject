"""Role 1: scoring a predicted line mask against the true line mask.

Both inputs are 0/1 arrays of the same size (1 = line).

Worked example (the one in our proposal):
    100 true line pixels, the model marks 80 pixels as lines, 60 of them right.
    TP = 60 (marked and really a line)
    FP = 20 (marked but not a line)
    FN = 40 (a real line the model missed)
    precision = TP / (TP + FP)      = 60 / 80          = 0.75
    recall    = TP / (TP + FN)      = 60 / 100         = 0.60
    dice      = 2TP / (2TP+FP+FN)   = 120 / 180        = 0.667
    iou       = TP / (TP + FP + FN) = 60 / 120         = 0.50

Tolerant scores: an artist's line and a predicted line can be the same line
but 1 pixel apart, which pixel-exact scores count as two mistakes. The
tolerant versions accept a predicted pixel if a true line is within
`tolerance` pixels, and a true pixel if a predicted line is within
`tolerance` pixels (like the standard edge-detection benchmark, BSDS).
"""

import cv2
import numpy as np


def _counts(pred, gt):
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    tp = int(np.logical_and(pred, gt).sum())
    fp = int(np.logical_and(pred, ~gt).sum())
    fn = int(np.logical_and(~pred, gt).sum())
    return tp, fp, fn


def _safe_div(a, b):
    # If both prediction and truth are empty, there is nothing to get wrong.
    return 1.0 if b == 0 else a / b


def dice(pred, gt):
    tp, fp, fn = _counts(pred, gt)
    return _safe_div(2 * tp, 2 * tp + fp + fn)


def iou(pred, gt):
    tp, fp, fn = _counts(pred, gt)
    return _safe_div(tp, tp + fp + fn)


def precision(pred, gt):
    tp, fp, _ = _counts(pred, gt)
    return _safe_div(tp, tp + fp)


def recall(pred, gt):
    tp, _, fn = _counts(pred, gt)
    return _safe_div(tp, tp + fn)


def _dilate(mask, tolerance):
    size = 2 * tolerance + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    return cv2.dilate(mask.astype(np.uint8), kernel) > 0


def tolerant_scores(pred, gt, tolerance=1):
    """Precision / recall / F1 that forgive lines off by <= tolerance pixels."""
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    if tolerance <= 0:
        p, r = precision(pred, gt), recall(pred, gt)
    else:
        p = _safe_div(int(np.logical_and(pred, _dilate(gt, tolerance)).sum()), int(pred.sum()))
        r = _safe_div(int(np.logical_and(gt, _dilate(pred, tolerance)).sum()), int(gt.sum()))
    f1 = 0.0 if p + r == 0 else 2 * p * r / (p + r)
    return {"tol_precision": p, "tol_recall": r, "tol_f1": f1}


def all_scores(pred, gt, tolerance=1):
    """Every number we report, for one image."""
    tp, fp, fn = _counts(pred, gt)
    scores = {
        "dice": _safe_div(2 * tp, 2 * tp + fp + fn),
        "iou": _safe_div(tp, tp + fp + fn),
        "precision": _safe_div(tp, tp + fp),
        "recall": _safe_div(tp, tp + fn),
    }
    scores.update(tolerant_scores(pred, gt, tolerance))
    return scores
