
import numpy as np
import cv2

def local_variance(img: np.ndarray, win: int = 9):
    """Compute local variance using box filters."""
    if img.dtype != np.float32:
        img = img.astype(np.float32)
    k = (win, win)
    mean = cv2.boxFilter(img, ddepth=-1, ksize=k, normalize=True, borderType=cv2.BORDER_REFLECT101)
    mean_sq = cv2.boxFilter(img*img, ddepth=-1, ksize=k, normalize=True, borderType=cv2.BORDER_REFLECT101)
    var = np.maximum(mean_sq - mean*mean, 0.0)
    return var

def adaptive_fusion(response_stack: np.ndarray, base_img: np.ndarray, var_windows=(7, 15, 31), gain=(1.0, 1.5, 2.0)):
    """
    Fuse orientation responses with weights derived from multi-scale local variance.
    - High-variance regions (likely veins) get higher gain.
    """
    if base_img.dtype != np.float32:
        base = base_img.astype(np.float32)/255.0
    else:
        base = base_img
    var_maps = [local_variance(base, w) for w in var_windows]
    # Normalize variance maps to [0,1]
    var_norm = [(v - v.min()) / (np.ptp(v) + 1e-8) for v in var_maps]
    # Build weight map as a convex combination with gains
    W = np.zeros_like(var_norm[0])
    total = 1e-8
    for v, g in zip(var_norm, gain):
        W += g * v
        total += g
    W = np.clip(W / total, 0, 1)
    # Enhance stack by per-pixel weight; also do max response
    max_r = response_stack.max(axis=-1)
    # Sharpen using unsharp mask guided by W
    blur = cv2.GaussianBlur(max_r, (0,0), sigmaX=1.0)
    sharp = np.clip(max_r + W * (max_r - blur), 0, 1)
    # Contrast stretch
    lo, hi = np.percentile(sharp, (2, 98))
    if hi - lo < 1e-6:
        stretched = sharp
    else:
        stretched = np.clip((sharp - lo) / (hi - lo), 0, 1)
    return stretched, W
