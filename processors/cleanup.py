"""
Morphological cleanup of the binary image after binarization.
Convention: text = 0 (black), background = 255 (white).
"""

import cv2
import numpy as np


def remove_small_components(binary: np.ndarray, min_area: int = 20) -> np.ndarray:
    """
    Delete connected components with area < min_area.
    Components touching the border are always kept (likely real text).
    """
    inv = cv2.bitwise_not(binary)   # text = 255 for connectedComponents
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        inv, connectivity=8
    )

    h, w = binary.shape
    keep_mask = np.zeros_like(inv)   # start all-zero: only add kept components

    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        lx   = int(stats[label, cv2.CC_STAT_LEFT])
        ly   = int(stats[label, cv2.CC_STAT_TOP])
        lw   = int(stats[label, cv2.CC_STAT_WIDTH])
        lh   = int(stats[label, cv2.CC_STAT_HEIGHT])

        touches_border = (lx == 0 or ly == 0 or
                          lx + lw >= w or ly + lh >= h)

        if area >= min_area or touches_border:
            keep_mask[labels == label] = 255

    # keep_mask = 255 where text is kept, 0 elsewhere
    # bitwise_not: kept text -> 0 (black), everywhere else -> 255 (white)
    return cv2.bitwise_not(keep_mask)


def close_stroke_gaps(binary: np.ndarray) -> np.ndarray:
    """Bridge 1-pixel gaps in ink strokes without merging distinct characters."""
    inv = cv2.bitwise_not(binary)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    inv = cv2.morphologyEx(inv, cv2.MORPH_CLOSE, kernel, iterations=1)
    return cv2.bitwise_not(inv)


def cleanup(binary: np.ndarray, min_component_area: int = 20) -> np.ndarray:
    """Remove small noise components only — no morphological thickening."""
    return remove_small_components(binary, min_area=min_component_area)
