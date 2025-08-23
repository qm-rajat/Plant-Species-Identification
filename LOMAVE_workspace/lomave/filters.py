
import numpy as np
import cv2
from typing import List, Tuple

def gabor_like_kernel(ksize: int, theta: float, sigma: float, lambd: float, gamma: float=0.5, psi: float=0.0):
    """Create a cosine-modulated Gaussian (Gabor-like) kernel with NumPy."""
    half = ksize // 2
    y, x = np.mgrid[-half:half+1, -half:half+1]
    # Rotate coordinates
    xr = x * np.cos(theta) + y * np.sin(theta)
    yr = -x * np.sin(theta) + y * np.cos(theta)
    gaussian = np.exp(-(xr**2 + (gamma**2)*(yr**2)) / (2*sigma**2))
    wave = np.cos(2*np.pi * xr / lambd + psi)
    kernel = gaussian * wave
    # zero-mean normalization
    kernel -= kernel.mean()
    norm = np.linalg.norm(kernel)
    if norm > 0:
        kernel /= norm
    return kernel.astype(np.float32)

def build_orientation_bank(n_orient: int = 12, ksize: int = 21, sigma: float = 3.0, lambd: float = 8.0, gamma: float = 0.5):
    thetas = [i * np.pi / n_orient for i in range(n_orient)]
    return [gabor_like_kernel(ksize, th, sigma, lambd, gamma) for th in thetas], thetas

def apply_filter_bank(img: np.ndarray, kernels: List[np.ndarray]) -> np.ndarray:
    """Convolve image with each kernel. Returns stack [H, W, K]."""
    if img.dtype != np.float32:
        img = img.astype(np.float32) / 255.0
    responses = []
    for k in kernels:
        r = cv2.filter2D(img, ddepth=-1, kernel=k, borderType=cv2.BORDER_REFLECT101)
        responses.append(r)
    stack = np.stack(responses, axis=-1)
    return stack

def max_response(stack: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return max response and argmax orientation index."""
    max_r = stack.max(axis=-1)
    ori_idx = stack.argmax(axis=-1)
    return max_r, ori_idx
