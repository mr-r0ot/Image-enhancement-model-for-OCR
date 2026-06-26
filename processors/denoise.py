"""
Noise removal for low-quality document images.

Three methods, from strongest to lightest:
- 'nlm'       : Non-local means — best quality, preserves strokes, slowest.
- 'bilateral' : Bilateral filter — fast edge-preserving smoothing.
- 'gaussian'  : Light Gaussian blur — only for very mild noise.
- 'none'      : Pass-through.
- 'auto'      : Picks method based on measured noise level (Laplacian variance).
"""

import cv2
import numpy as np


def _noise_level(gray: np.ndarray) -> float:
    """Estimate noise via Laplacian variance (proxy for high-frequency energy)."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def denoise_nlm(gray: np.ndarray, h: float = 10.0) -> np.ndarray:
    return cv2.fastNlMeansDenoising(
        gray, None,
        h=h,
        templateWindowSize=7,
        searchWindowSize=21,
    )


def denoise_bilateral(gray: np.ndarray,
                       d: int = 9,
                       sigma_color: float = 75.0,
                       sigma_space: float = 75.0) -> np.ndarray:
    return cv2.bilateralFilter(gray, d, sigma_color, sigma_space)


def denoise_gaussian(gray: np.ndarray, ksize: int = 3) -> np.ndarray:
    return cv2.GaussianBlur(gray, (ksize, ksize), 0)


def denoise(gray: np.ndarray, method: str = 'auto', **kw) -> np.ndarray:
    """
    Denoise grayscale image.

    kw keys forwarded from EnhancementConfig:
        nlm_h, bilateral_d, bilateral_sigma_color, bilateral_sigma_space
    """
    if method == 'none':
        return gray.copy()

    if method == 'nlm':
        return denoise_nlm(gray, h=kw.get('nlm_h', 10.0))

    if method == 'bilateral':
        return denoise_bilateral(
            gray,
            d=kw.get('bilateral_d', 9),
            sigma_color=kw.get('bilateral_sigma_color', 75.0),
            sigma_space=kw.get('bilateral_sigma_space', 75.0),
        )

    if method == 'gaussian':
        return denoise_gaussian(gray)

    # auto
    noise = _noise_level(gray)
    if noise > 600:
        return denoise_nlm(gray, h=kw.get('nlm_h', 13.0))
    elif noise > 200:
        return denoise_nlm(gray, h=kw.get('nlm_h', 8.0))
    elif noise > 80:
        return denoise_bilateral(
            gray,
            d=kw.get('bilateral_d', 9),
            sigma_color=kw.get('bilateral_sigma_color', 75.0),
            sigma_space=kw.get('bilateral_sigma_space', 75.0),
        )
    return gray.copy()
