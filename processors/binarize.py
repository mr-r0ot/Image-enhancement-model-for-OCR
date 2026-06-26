"""
Binarization for handwritten document images.

Methods:
- 'sauvola'  : Sauvola adaptive thresholding -- gold standard for handwriting.
- 'otsu'     : Global Otsu threshold.
- 'adaptive' : OpenCV adaptive Gaussian threshold.
- 'none'     : Skip (keep grayscale).
- 'auto'     : Defaults to Sauvola.

Output convention: text = black (0), background = white (255).
"""

import cv2
import numpy as np


def _ensure_dark_text(binary: np.ndarray) -> np.ndarray:
    """Guarantee text pixels are black (minority pixels in a document)."""
    if np.mean(binary == 0) > 0.5:
        return cv2.bitwise_not(binary)
    return binary


def binarize_sauvola(gray: np.ndarray,
                      window_size: int = 25,
                      k: float = 0.15,
                      R: float = 128.0) -> np.ndarray:
    """
    Sauvola binarization.
    Threshold surface is computed on a lightly blurred copy to avoid
    scatter artifacts from noise, but the original pixel values are
    compared against that threshold so stroke shapes are preserved.
    """
    if window_size % 2 == 0:
        window_size += 1

    gf = gray.astype(np.float64)
    # Smooth copy for stable threshold surface — reduces salt-pepper scatter
    gf_s = cv2.GaussianBlur(gf, (0, 0), 0.8)

    mean = cv2.boxFilter(gf_s, ddepth=cv2.CV_64F,
                         ksize=(window_size, window_size),
                         normalize=True,
                         borderType=cv2.BORDER_REFLECT)
    mean_sq = cv2.boxFilter(gf_s * gf_s, ddepth=cv2.CV_64F,
                            ksize=(window_size, window_size),
                            normalize=True,
                            borderType=cv2.BORDER_REFLECT)

    variance = np.maximum(mean_sq - mean * mean, 0.0)
    std = np.sqrt(variance)

    threshold = mean * (1.0 + k * (std / R - 1.0))
    # Compare original (unblurred) pixel to the smooth threshold
    binary = np.where(gf < threshold, 0, 255).astype(np.uint8)
    return _ensure_dark_text(binary)


def binarize_otsu(gray: np.ndarray) -> np.ndarray:
    _, binary = cv2.threshold(gray, 0, 255,
                               cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return _ensure_dark_text(binary)


def binarize_adaptive(gray: np.ndarray,
                       block_size: int = 35,
                       C: int = 10) -> np.ndarray:
    if block_size % 2 == 0:
        block_size += 1
    binary = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size, C,
    )
    return _ensure_dark_text(binary)


def binarize(gray: np.ndarray,
             method: str = 'sauvola',
             sauvola_window: int = 25,
             sauvola_k: float = 0.2) -> np.ndarray:
    """
    Binarize grayscale image.
    Returns uint8 image: text = 0 (black), background = 255 (white).
    """
    if method == 'none':
        return gray.copy()
    if method in ('sauvola', 'auto'):
        return binarize_sauvola(gray, window_size=sauvola_window, k=sauvola_k)
    if method == 'otsu':
        return binarize_otsu(gray)
    if method == 'adaptive':
        return binarize_adaptive(gray)
    return gray.copy()
