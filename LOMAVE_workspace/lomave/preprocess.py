
import cv2
import numpy as np

def to_grayscale(img):
    if img.ndim == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def denoise(img, method="gaussian", ksize=5, sigma=1.0):
    if method == "median":
        return cv2.medianBlur(img, ksize)
    # default gaussian
    return cv2.GaussianBlur(img, (ksize, ksize), sigmaX=sigma, sigmaY=sigma)

def equalize(img, method="clahe", clip=2.0, tile_grid_size=(8,8)):
    if method == "hist":
        return cv2.equalizeHist(img)
    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=tile_grid_size)
    return clahe.apply(img)

def preprocess(img_bgr, denoise_method="gaussian", ksize=5, sigma=1.0, eq_method="clahe"):
    gray = to_grayscale(img_bgr)
    gray = denoise(gray, method=denoise_method, ksize=ksize, sigma=sigma)
    gray = equalize(gray, method=eq_method)
    return gray
