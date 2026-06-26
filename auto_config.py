"""
Analyzes an input image and returns the optimal EnhancementConfig.
All decisions are based on measurable image quality metrics.
"""

import cv2
import numpy as np
from config import EnhancementConfig


def analyze_image(gray: np.ndarray) -> dict:
    h, w = gray.shape
    min_dim = min(h, w)

    # Noise level via Laplacian variance (< 100 clean, 100-500 moderate, > 500 noisy)
    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    # Lighting uniformity
    blur_k = max(51, (min_dim // 20) | 1)
    blurred = cv2.GaussianBlur(gray.astype(np.float32), (blur_k, blur_k), 0)
    mean_lum = float(np.mean(blurred))
    lighting_variation = float(np.std(blurred)) / (mean_lum + 1e-6)

    # Contrast (2nd-98th percentile range)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    hist /= hist.sum()
    cumsum = np.cumsum(hist)
    low  = int(np.searchsorted(cumsum, 0.02))
    high = int(np.searchsorted(cumsum, 0.98))
    contrast_range = high - low

    # Rotation skew via Hough lines
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180,
        threshold=max(40, min_dim // 25),
        minLineLength=min_dim // 8,
        maxLineGap=min_dim // 20,
    )

    skew_angle = 0.0
    has_skew = False
    if lines is not None and len(lines) >= 3:
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if abs(x2 - x1) < 3:
                continue
            a = float(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
            if a > 45:
                a -= 90
            elif a < -45:
                a += 90
            if abs(a) <= 45:
                angles.append(a)
        if len(angles) >= 3:
            skew_angle = float(np.median(angles))
            has_skew = abs(skew_angle) > 1.0

    return {
        'noise_level': laplacian_var,
        'lighting_variation': lighting_variation,
        'contrast_range': contrast_range,
        'mean_luminance': mean_lum,
        'min_dimension': min_dim,
        'skew_angle': skew_angle,
        'has_skew': has_skew,
    }


def auto_configure(gray: np.ndarray, user_overrides: dict = None):
    """
    Analyze image and build optimal EnhancementConfig.
    Returns (config, metrics).
    """
    metrics = analyze_image(gray)
    cfg = EnhancementConfig()

    min_dim = metrics['min_dimension']
    noise   = metrics['noise_level']
    lv      = metrics['lighting_variation']
    contrast = metrics['contrast_range']
    h, w = gray.shape

    # --- Upscale ---
    if min_dim >= 1500:
        cfg.upscale = False
    elif min_dim < 400:
        cfg.upscale = True
        cfg.target_min_dimension = 1200
        cfg.max_upscale_factor = 3.0
    elif min_dim < 900:
        cfg.upscale = True
        cfg.target_min_dimension = 1500
        cfg.max_upscale_factor = 2.5
    else:
        cfg.upscale = True
        cfg.target_min_dimension = 1500
        cfg.max_upscale_factor = 2.0

    # --- Deskew ---
    cfg.deskew = True
    cfg.deskew_method = 'auto'

    # --- Background normalization ---
    cfg.normalize_background = lv > 0.08 or contrast < 150

    # --- CLAHE ---
    cfg.clahe = True
    if lv > 0.35:
        cfg.clahe_clip_limit = 3.5
    elif lv > 0.18:
        cfg.clahe_clip_limit = 2.5
    else:
        cfg.clahe_clip_limit = 2.0

    # --- Denoising ---
    # Light bilateral only for very noisy images; preserve ink edges.
    if noise > 400:
        cfg.denoise = True
        cfg.denoise_method = 'bilateral'
        cfg.bilateral_d = 7
        cfg.bilateral_sigma_color = 50.0
        cfg.bilateral_sigma_space = 50.0
    else:
        cfg.denoise = False

    # --- Binarization ---
    cfg.binarize = True
    cfg.binarize_method = 'sauvola'

    # Larger windows = more stable threshold = less scatter
    if min_dim < 500:
        cfg.sauvola_window = 21
    elif min_dim < 1000:
        cfg.sauvola_window = 31
    elif min_dim < 2000:
        cfg.sauvola_window = 41
    else:
        cfg.sauvola_window = 51

    # Low k = fewer false positives = cleaner, thinner strokes
    cfg.sauvola_k = 0.10

    # --- Morphological cleanup ---
    cfg.morph_cleanup = True
    cfg.min_component_area = max(8, int(h * w * 0.000008))

    # --- Apply user overrides ---
    if user_overrides:
        cfg = cfg.apply_overrides(user_overrides)

    return cfg, metrics
