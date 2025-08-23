
import cv2
import numpy as np
from skimage.filters import sobel, prewitt, laplace
from scipy.ndimage import gaussian_laplace

def sobel_edges(gray):
    if gray.dtype != np.float32:
        g = gray.astype(np.float32)/255.0
    else:
        g = gray
    e = sobel(g)
    return (e*255).astype(np.uint8)

def prewitt_edges(gray):
    if gray.dtype != np.float32:
        g = gray.astype(np.float32)/255.0
    else:
        g = gray
    e = prewitt(g)
    return (e*255).astype(np.uint8)

def canny_edges(gray, low=50, high=150):
    return cv2.Canny(gray, low, high)

def log_edges(gray, sigma=1.0):
    if gray.dtype != np.float32:
        g = gray.astype(np.float32)/255.0
    else:
        g = gray
    e = gaussian_laplace(g, sigma=sigma)
    e = np.abs(e)
    e = (e / (e.max() + 1e-8) * 255).astype(np.uint8)
    return e
