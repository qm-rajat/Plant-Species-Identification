
import numpy as np
from skimage.metrics import structural_similarity as ssim

def binarize(img, thresh=0.5):
    if img.dtype != np.float32:
        img = img.astype(np.float32)
        if img.max() > 1.1:
            img = img / 255.0
    return (img >= thresh).astype(np.uint8)

def confusion(pred: np.ndarray, gt: np.ndarray):
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    tp = (pred & gt).sum()
    tn = ((~pred) & (~gt)).sum()
    fp = (pred & (~gt)).sum()
    fn = ((~pred) & gt).sum()
    return tp, tn, fp, fn

def classification_metrics(pred_bin: np.ndarray, gt_bin: np.ndarray):
    tp, tn, fp, fn = confusion(pred_bin, gt_bin)
    acc = (tp + tn) / max(tp + tn + fp + fn, 1)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2*prec*rec / max(prec + rec, 1e-8)
    return {"accuracy": float(acc), "precision": float(prec), "recall": float(rec), "f1": float(f1)}

def ssim_score(img1: np.ndarray, img2: np.ndarray):
    # Expect uint8 or float in [0,1]
    if img1.dtype != img2.dtype:
        img1 = img1.astype(np.float32)
        img2 = img2.astype(np.float32)
    if img1.max() > 1.1:
        img1 = img1/255.0
    if img2.max() > 1.1:
        img2 = img2/255.0
    return float(ssim(img1, img2, data_range=1.0))
