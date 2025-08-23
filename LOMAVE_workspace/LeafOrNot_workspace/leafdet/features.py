
import cv2
import numpy as np
from skimage.measure import regionprops, label
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops

def compute_shape_features(mask):
    h, w = mask.shape
    area_img = float(h*w)
    lbl = label(mask>0)
    props = regionprops(lbl)
    if not props:
        return dict(area_ratio=0.0, solidity=0.0, aspect_ratio=0.0, circularity=0.0)
    p = max(props, key=lambda x: x.area)
    area_ratio = p.area / area_img
    solidity = p.solidity if hasattr(p, "solidity") else p.area / max(p.convex_area, 1)
    minr, minc, maxr, maxc = p.bbox
    aspect_ratio = (maxc-minc) / max(maxr-minr, 1)
    perim = p.perimeter if p.perimeter > 0 else 1.0
    circularity = 4*np.pi*(p.area/(perim**2))
    return dict(area_ratio=float(area_ratio), solidity=float(solidity),
                aspect_ratio=float(aspect_ratio), circularity=float(circularity))

def compute_color_features(img_bgr, mask=None):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    if mask is not None and mask.any():
        m = mask.astype(bool)
        hs = hsv[m]
        ls = lab[m]
    else:
        hs = hsv.reshape(-1,3)
        ls = lab.reshape(-1,3)
    feats = {
        "h_mean": float(hs[:,0].mean()), "h_std": float(hs[:,0].std()),
        "s_mean": float(hs[:,1].mean()), "s_std": float(hs[:,1].std()),
        "v_mean": float(hs[:,2].mean()), "v_std": float(hs[:,2].std()),
        "a_mean": float(ls[:,1].mean()), "a_std": float(ls[:,1].std()),
        "b_mean": float(ls[:,2].mean()), "b_std": float(ls[:,2].std())
    }
    return feats

def compute_texture_features(gray, mask=None, lbp_radius=2, lbp_points=16, distances=(1,2), angles=(0, np.pi/4, np.pi/2, 3*np.pi/4)):
    if gray.max() > 1.0:
        g = gray.astype(np.uint8)
    else:
        g = (gray*255).astype(np.uint8)
    if mask is not None and mask.any():
        g = g.copy()
        g[~mask.astype(bool)] = 0
    lbp = local_binary_pattern(g, P=lbp_points, R=lbp_radius, method="uniform")
    n_bins = lbp_points + 2
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
    feats = {f"lbp_{i}": float(h) for i, h in enumerate(hist.tolist())}
    glcm = graycomatrix(g, distances=distances, angles=angles, levels=256, symmetric=True, normed=True)
    for prop in ["contrast", "homogeneity", "energy", "correlation", "ASM", "dissimilarity"]:
        try:
            vals = graycoprops(glcm, prop)
            feats[f"glcm_{prop}_mean"] = float(vals.mean())
        except Exception:
            pass
    return feats
