
import os, glob, json
import cv2
import numpy as np
from .preprocess import resize_max, to_gray
from .segment import green_mask_hsv, excess_green_mask, largest_component, clean_mask
from .features import compute_shape_features, compute_color_features, compute_texture_features

def features_for_image(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(path)
    img = resize_max(img, 640)
    gray = to_gray(img)
    mask = green_mask_hsv(img) | excess_green_mask(img)
    mask = clean_mask(mask, 5)
    comp = largest_component(mask, min_area=int(0.02*img.shape[0]*img.shape[1]))
    feats = {}
    feats.update(compute_shape_features(comp))
    feats.update(compute_color_features(img, comp))
    feats.update(compute_texture_features(gray, comp))
    return feats

def build_matrix_from_folders(leaf_dir, nonleaf_dir):
    X, y, files = [], [], []
    feat_names = None
    for label, d in [(1, leaf_dir), (0, nonleaf_dir)]:
        for p in glob.glob(os.path.join(d, "*.*")):
            if p.lower().endswith((".png",".jpg",".jpeg",".bmp",".tif",".tiff")):
                try:
                    feats = features_for_image(p)
                    if feat_names is None:
                        feat_names = sorted(feats.keys())
                    X.append([feats[k] for k in feat_names])
                    y.append(label)
                    files.append(p)
                except Exception as e:
                    print(f"[WARN] {p}: {e}")
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32), files, feat_names or []
