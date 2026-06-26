"""
Skew detection and correction for document images.

Two methods:
- 'hough'      : Fast Hough-line based angle estimation.
- 'projection' : Projection-profile variance search. Slower but more robust
                 for heavily skewed pages with few long straight lines.
- 'auto'       : Hough first; if the result looks uncertain, confirm with
                 a narrow projection search around the Hough estimate.
"""

import cv2
import numpy as np


# ── helpers ──────────────────────────────────────────────────────────────────

def _otsu_binary(gray: np.ndarray) -> np.ndarray:
    _, binary = cv2.threshold(gray, 0, 255,
                               cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return binary


def _rotate(gray: np.ndarray, angle: float, bg_color: int = 255) -> np.ndarray:
    """Rotate image, expanding canvas so no content is clipped."""
    h, w = gray.shape
    cx, cy = w / 2.0, h / 2.0
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)

    cos_a = abs(M[0, 0])
    sin_a = abs(M[0, 1])
    new_w = int(h * sin_a + w * cos_a)
    new_h = int(h * cos_a + w * sin_a)

    M[0, 2] += (new_w - w) / 2.0
    M[1, 2] += (new_h - h) / 2.0

    return cv2.warpAffine(gray, M, (new_w, new_h),
                          flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_CONSTANT,
                          borderValue=bg_color)


# ── methods ───────────────────────────────────────────────────────────────────

def _hough_angle(gray: np.ndarray, max_angle: float) -> tuple:
    """
    Returns (angle_degrees, confidence 0-1).
    confidence is based on how many lines agreed on the angle.
    """
    h, w = gray.shape
    min_dim = min(h, w)

    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 360,
        threshold=max(30, min_dim // 30),
        minLineLength=min_dim // 8,
        maxLineGap=min_dim // 20,
    )

    if lines is None or len(lines) < 3:
        return 0.0, 0.0

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
        if abs(a) <= max_angle:
            angles.append(a)

    if len(angles) < 3:
        return 0.0, 0.0

    angle = float(np.median(angles))
    # Confidence: fraction of angles within 5° of median
    close = sum(1 for a in angles if abs(a - angle) < 5.0)
    confidence = close / len(angles)
    return angle, confidence


def _projection_angle(gray: np.ndarray, search_center: float,
                       search_range: float, step: float) -> float:
    """
    Sweep angles in [search_center - search_range, search_center + search_range]
    and return angle that maximises horizontal projection variance.
    Downsamples to ≤ 600 px on longest side for speed.
    """
    h, w = gray.shape
    scale = min(1.0, 600.0 / max(h, w))
    if scale < 0.99:
        small = cv2.resize(gray, None, fx=scale, fy=scale,
                           interpolation=cv2.INTER_AREA)
    else:
        small = gray

    binary = _otsu_binary(small)
    sh, sw = binary.shape
    cx, cy = sw / 2.0, sh / 2.0

    best_angle = search_center
    best_var = -1.0

    a = search_center - search_range
    while a <= search_center + search_range + 1e-9:
        M = cv2.getRotationMatrix2D((cx, cy), a, 1.0)
        rotated = cv2.warpAffine(binary, M, (sw, sh),
                                  flags=cv2.INTER_NEAREST,
                                  borderMode=cv2.BORDER_CONSTANT,
                                  borderValue=0)
        proj = rotated.sum(axis=1).astype(np.float64)
        v = float(np.var(proj))
        if v > best_var:
            best_var = v
            best_angle = a
        a += step

    return best_angle


# ── public API ────────────────────────────────────────────────────────────────

def deskew(gray: np.ndarray,
           method: str = 'auto',
           max_angle: float = 45.0,
           angle_step: float = 0.5) -> tuple:
    """
    Correct skew in a grayscale document image.

    Returns
    -------
    (deskewed_image, corrected_angle_degrees)
    corrected_angle is the angle that was applied (positive = counter-clockwise).
    """
    # Estimate background color for padding
    bg_color = int(np.percentile(gray, 95))

    if method == 'hough':
        angle, _ = _hough_angle(gray, max_angle)

    elif method == 'projection':
        angle = _projection_angle(gray, 0.0, max_angle, angle_step)

    else:  # 'auto'
        hough_angle, confidence = _hough_angle(gray, max_angle)

        if confidence >= 0.6 and abs(hough_angle) >= 0.3:
            # High-confidence Hough result — use it directly
            angle = hough_angle
        elif abs(hough_angle) >= 0.3:
            # Low confidence: verify with projection in a narrow window
            search_range = min(10.0, max_angle)
            angle = _projection_angle(gray, hough_angle, search_range, angle_step)
        else:
            # Hough found ~0° — do a broad projection sweep to confirm
            angle = _projection_angle(gray, 0.0, max_angle, angle_step)

    if abs(angle) < 0.3:
        return gray.copy(), 0.0

    deskewed = _rotate(gray, angle, bg_color)
    return deskewed, angle
