import copy
from dataclasses import dataclass
from typing import Literal


@dataclass
class EnhancementConfig:
    # --- Resize ---
    upscale: bool = True
    target_min_dimension: int = 1500
    max_upscale_factor: float = 3.0

    # --- Deskew ---
    deskew: bool = True
    deskew_method: Literal['projection', 'hough', 'auto'] = 'auto'
    max_skew_angle: float = 45.0
    deskew_angle_step: float = 0.5

    # --- Background normalization ---
    normalize_background: bool = True
    bg_kernel_ratio: float = 0.08

    # --- Illumination (CLAHE) ---
    clahe: bool = True
    clahe_clip_limit: float = 2.0
    clahe_grid_size: int = 8

    # --- Denoising ---
    denoise: bool = True
    denoise_method: Literal['nlm', 'bilateral', 'gaussian', 'none', 'auto'] = 'auto'
    nlm_h: float = 10.0
    bilateral_d: int = 9
    bilateral_sigma_color: float = 75.0
    bilateral_sigma_space: float = 75.0

    # --- Binarization ---
    binarize: bool = True
    binarize_method: Literal['sauvola', 'otsu', 'adaptive', 'none', 'auto'] = 'auto'
    sauvola_window: int = 25
    sauvola_k: float = 0.15

    # --- Morphological cleanup ---
    morph_cleanup: bool = True
    min_component_area: int = 0   # 0 = auto-compute based on image size

    # --- Output ---
    invert_output: bool = False

    def apply_overrides(self, overrides: dict) -> 'EnhancementConfig':
        cfg = copy.copy(self)
        for k, v in overrides.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
        return cfg
