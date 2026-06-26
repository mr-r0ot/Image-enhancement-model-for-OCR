"""
Illumination correction for document images with uneven lighting.

Steps applied here:
1. Background estimation via morphological operations.
   - Dilation removes dark text strokes → gives the local background level.
   - Dividing the image by this estimate flattens uneven illumination.
2. CLAHE for local contrast enhancement (handles remaining local contrast issues).
"""

import cv2
import numpy as np


def _is_dark_text(gray: np.ndarray) -> bool:
    """Return True if text is dark on a light background (the common case)."""
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Dark pixels are minority in a document with dark text on white paper
    return float(np.sum(otsu == 0)) < float(np.sum(otsu == 255))


def normalize_background(gray: np.ndarray, kernel_ratio: float = 0.08) -> np.ndarray:
    """
    Flatten background illumination using morphological background estimation.

    kernel_ratio: kernel size as fraction of min(height, width).
                  Larger = handles bigger illumination gradients but slower.
    """
    h, w = gray.shape
    k = max(21, int(min(h, w) * kernel_ratio))
    if k % 2 == 0:
        k += 1

    dark_text = _is_dark_text(gray)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

    if dark_text:
        # Dilation expands bright regions → replaces dark text with local max → background
        background = cv2.dilate(gray, kernel)
    else:
        # Erosion shrinks bright regions → background for light-text-on-dark
        background = cv2.erode(gray, kernel)

    # Smooth to avoid block artefacts from the kernel
    background = cv2.GaussianBlur(background.astype(np.float32),
                                   (k, k), sigmaX=k / 3.0)

    gray_f = gray.astype(np.float32)

    if dark_text:
        # normalised = image / background → text stays dark, background → 1.0
        normalized = (gray_f / (background + 1e-6)) * 255.0
    else:
        # For inverted documents: normalise by how much brighter text is than bg
        normalized = ((background - gray_f) / (background + 1e-6)) * 255.0

    normalized = np.clip(normalized, 0, 255).astype(np.uint8)

    # Ensure result still has dark text (sometimes the normalisation can flip polarity)
    if _is_dark_text(normalized) != dark_text:
        normalized = cv2.bitwise_not(normalized)

    return normalized


def apply_clahe(gray: np.ndarray,
                clip_limit: float = 2.0,
                grid_size: int = 8) -> np.ndarray:
    """CLAHE — boosts local contrast without over-amplifying noise."""
    clahe = cv2.createCLAHE(clipLimit=clip_limit,
                              tileGridSize=(grid_size, grid_size))
    return clahe.apply(gray)
