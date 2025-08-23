
import cv2
import numpy as np

def load_image(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img

def to_gray(img):
    if img.ndim == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def to_hsv(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

def resize_max(img, max_side=640):
    h, w = img.shape[:2]
    scale = max(h, w) / float(max_side)
    if scale <= 1.0:
        return img
    nh, nw = int(h/scale), int(w/scale)
    return cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)

def denoise(img, ksize=3):
    m = cv2.medianBlur(img, ksize)
    g = cv2.GaussianBlur(m, (ksize|1, ksize|1), 0)
    return g

def clahe_gray(gray, clip=2.0, grid=(8,8)):
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=grid)
    return clahe.apply(gray)
