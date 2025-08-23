
import numpy as np
import cv2
from skimage.morphology import skeletonize, remove_small_objects
from skimage.measure import label, regionprops

def binarize(enhanced: np.ndarray, thresh_method: str = "otsu"):
    img8 = (np.clip(enhanced,0,1) * 255).astype(np.uint8)
    if thresh_method == "otsu":
        _, th = cv2.threshold(img8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        # adaptive
        th = cv2.adaptiveThreshold(img8, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 35, -2)
    return th

def skeletonize_veins(binary: np.ndarray, min_size: int = 30):
    # Ensure foreground veins are 1
    bw = (binary > 0).astype(np.uint8)
    # Remove small specks
    lbl = label(bw, connectivity=2)
    bw_clean = remove_small_objects(lbl, min_size=min_size)
    bw_clean = (bw_clean > 0).astype(np.uint8)
    skel = skeletonize(bw_clean).astype(np.uint8)
    return skel

def leaf_mask_from_image(gray_img: np.ndarray):
    # Simple Otsu for leaf region (assumes leaf darker than background)
    img8 = (gray_img if gray_img.dtype==np.uint8 else (np.clip(gray_img,0,1)*255).astype(np.uint8))
    _, th = cv2.threshold(img8, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # Choose largest connected component as leaf
    lbl = label(th > 0)
    areas = [(r.area, r.label) for r in regionprops(lbl)]
    if not areas:
        return (th>0).astype(np.uint8)
    largest = max(areas)[1]
    mask = (lbl == largest).astype(np.uint8)
    return mask
