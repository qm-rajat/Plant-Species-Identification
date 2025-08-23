
import cv2
import numpy as np
from .segment import green_mask_hsv, excess_green_mask, largest_component, clean_mask
from .features import compute_shape_features, compute_color_features

def quick_leaf_rule(img_bgr):
    h, w = img_bgr.shape[:2]
    area_img = float(h*w)
    mask_hsv = green_mask_hsv(img_bgr)
    mask_exg = excess_green_mask(img_bgr)
    mask = ((mask_hsv > 0) | (mask_exg > 0)).astype(np.uint8) * 255
    mask = clean_mask(mask, k=5)
    comp = largest_component(mask, min_area=int(0.02*area_img))
    green_ratio = mask.sum() / (255.0*area_img)
    comp_ratio = comp.sum() / float(area_img)
    shape = compute_shape_features(comp)
    color = compute_color_features(img_bgr, comp)
    score = 0.0
    score += min(green_ratio / 0.30, 1.0) * 0.4
    score += min(comp_ratio / 0.10, 1.0) * 0.2
    score += min(max(shape["solidity"], 0.0), 1.0) * 0.2
    score += min(shape["circularity"] / 0.6, 1.0) * 0.2
    label = "leaf" if score >= 0.55 else "non-leaf"
    diag = {
        "green_ratio": float(green_ratio),
        "component_ratio": float(comp_ratio),
        "shape": shape,
        "color_means": {k:v for k,v in color.items() if k.endswith("_mean")},
        "score": float(score)
    }
    return label, score, diag
