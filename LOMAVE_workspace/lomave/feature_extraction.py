
import numpy as np
from skimage.measure import label, regionprops
from scipy.ndimage import convolve

# 8-neighborhood offsets
K4 = np.array([[0,1,0],
               [1,0,1],
               [0,1,0]], dtype=np.uint8)
K8 = np.array([[1,1,1],
               [1,0,1],
               [1,1,1]], dtype=np.uint8)

def vein_density(skeleton: np.ndarray, leaf_mask: np.ndarray):
    area = max(leaf_mask.sum(), 1)
    veins = skeleton.sum()
    return veins / area

def skeleton_length_px(skeleton: np.ndarray):
    """Approximate geodesic length by counting 4- and 8-connected steps."""
    sk = (skeleton > 0).astype(np.uint8)
    n4 = ((convolve(sk, K4, mode='constant') > 0) & (sk==1)).sum()
    n8 = ((convolve(sk, K8, mode='constant') > 0) & (sk==1)).sum() - n4
    # Approximate length: 4-neighbors -> 1, diagonal -> sqrt(2)
    return n4 * 1.0 + n8 * np.sqrt(2)

def vein_to_area_ratio(skeleton: np.ndarray, leaf_mask: np.ndarray):
    area = max(leaf_mask.sum(), 1)
    length = skeleton_length_px(skeleton)
    return length / area

def extract_features(skeleton: np.ndarray, leaf_mask: np.ndarray):
    return {
        "vein_density": float(vein_density(skeleton, leaf_mask)),
        "skeleton_length_px": float(skeleton_length_px(skeleton)),
        "vein_to_area_ratio": float(vein_to_area_ratio(skeleton, leaf_mask))
    }
