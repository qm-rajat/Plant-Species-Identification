
import cv2
import numpy as np
from skimage.measure import label, regionprops

def green_mask_hsv(img_bgr, h_low=25, h_high=95, s_low=40, v_low=40):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lower = np.array([h_low, s_low, v_low], dtype=np.uint8)
    upper = np.array([h_high, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    return mask

def excess_green_mask(img_bgr, thresh=20):
    b, g, r = cv2.split(img_bgr.astype(np.int16))
    exg = 2*g - r - b
    exg = np.clip(exg, 0, 255).astype(np.uint8)
    _, mask = cv2.threshold(exg, thresh, 255, cv2.THRESH_BINARY)
    return mask

def largest_component(mask, min_area=200):
    bw = (mask > 0).astype(np.uint8)
    lbl = label(bw, connectivity=2)
    props = regionprops(lbl)
    if len(props) == 0:
        return np.zeros_like(bw, dtype=np.uint8)
    largest = max(props, key=lambda p: p.area)
    if largest.area < min_area:
        return np.zeros_like(bw, dtype=np.uint8)
    result = (lbl == largest.label).astype(np.uint8)
    return result

def clean_mask(mask, k=5):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.medianBlur(mask, 5)
    return (mask>0).astype(np.uint8)
