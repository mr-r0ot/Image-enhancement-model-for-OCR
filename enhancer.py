"""
Main pipeline that chains all processors together.
"""

import cv2
import numpy as np
from pathlib import Path

from config import EnhancementConfig
from auto_config import auto_configure
from processors.deskew import deskew
from processors.illumination import normalize_background, apply_clahe
from processors.denoise import denoise
from processors.binarize import binarize
from processors.cleanup import cleanup


class ImageEnhancer:
    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    def _log(self, msg: str):
        if self.verbose:
            print(f"  [enhance] {msg}")

    def _load(self, path: str) -> np.ndarray:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {path}")
        return img

    def _to_gray(self, img: np.ndarray) -> np.ndarray:
        if len(img.shape) == 3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img.copy()

    def _upscale(self, gray: np.ndarray, cfg: EnhancementConfig) -> np.ndarray:
        h, w = gray.shape
        min_dim = min(h, w)
        if not cfg.upscale or min_dim >= cfg.target_min_dimension:
            return gray
        scale = min(cfg.target_min_dimension / min_dim, cfg.max_upscale_factor)
        if scale <= 1.01:
            return gray
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
        self._log(f"Upscale {w}x{h} -> {new_w}x{new_h} ({scale:.2f}x)")
        return cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    def process(self, input_path: str, output_path: str,
                user_overrides: dict = None) -> dict:
        """
        Run the full enhancement pipeline on one image.
        Returns dict: success, output_path, config, metrics, error.
        """
        user_overrides = user_overrides or {}

        try:
            # 1. Load
            self._log(f"Loading  {input_path}")
            img  = self._load(input_path)
            gray = self._to_gray(img)
            h0, w0 = gray.shape
            self._log(f"Input    {w0}x{h0} px")

            # 2. Auto-analyse & build config
            self._log("Analysing image...")
            cfg, metrics = auto_configure(gray, user_overrides)
            self._log(
                f"Metrics  noise={metrics['noise_level']:.0f}  "
                f"light_var={metrics['lighting_variation']:.3f}  "
                f"contrast={metrics['contrast_range']}  "
                f"skew={metrics['skew_angle']:.1f}deg"
            )
            self._log(
                f"Config   upscale={cfg.upscale}  deskew={cfg.deskew}  "
                f"bg_norm={cfg.normalize_background}  clahe={cfg.clahe}  "
                f"denoise={cfg.denoise_method}  "
                f"binarize={cfg.binarize_method}(w={cfg.sauvola_window},k={cfg.sauvola_k})"
            )

            # 3. Upscale
            gray = self._upscale(gray, cfg)

            # 4. Deskew
            if cfg.deskew:
                gray, angle = deskew(
                    gray,
                    method=cfg.deskew_method,
                    max_angle=cfg.max_skew_angle,
                    angle_step=cfg.deskew_angle_step,
                )
                self._log(f"Deskew   corrected {angle:+.1f}deg")

            # 5. Background normalisation
            if cfg.normalize_background:
                self._log("BG-norm  ...")
                gray = normalize_background(gray, kernel_ratio=cfg.bg_kernel_ratio)

            # 6. CLAHE
            if cfg.clahe:
                self._log(f"CLAHE    clip={cfg.clahe_clip_limit}")
                gray = apply_clahe(gray,
                                    clip_limit=cfg.clahe_clip_limit,
                                    grid_size=cfg.clahe_grid_size)

            # 7. Denoise
            if cfg.denoise and cfg.denoise_method != 'none':
                self._log(f"Denoise  {cfg.denoise_method}")
                gray = denoise(
                    gray,
                    method=cfg.denoise_method,
                    nlm_h=cfg.nlm_h,
                    bilateral_d=cfg.bilateral_d,
                    bilateral_sigma_color=cfg.bilateral_sigma_color,
                    bilateral_sigma_space=cfg.bilateral_sigma_space,
                )

            # 8. Binarize
            if cfg.binarize and cfg.binarize_method != 'none':
                self._log(
                    f"Binarize {cfg.binarize_method}  "
                    f"k={cfg.sauvola_k}  w={cfg.sauvola_window}"
                )
                gray = binarize(
                    gray,
                    method=cfg.binarize_method,
                    sauvola_window=cfg.sauvola_window,
                    sauvola_k=cfg.sauvola_k,
                )

            # 9. Morphological cleanup
            if cfg.morph_cleanup and cfg.binarize and cfg.binarize_method != 'none':
                min_area = cfg.min_component_area or max(
                    8, int(gray.shape[0] * gray.shape[1] * 0.000008)
                )
                self._log(f"Cleanup  min_area={min_area}px")
                gray = cleanup(gray, min_component_area=min_area)

            # 10. Optional invert
            if cfg.invert_output:
                gray = cv2.bitwise_not(gray)

            # 11. Save
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            ok = cv2.imwrite(output_path, gray)
            if not ok:
                raise IOError(f"cv2.imwrite failed: {output_path}")

            h1, w1 = gray.shape
            self._log(f"Saved    {output_path}  ({w1}x{h1} px)")

            return {
                'success': True,
                'output_path': output_path,
                'config': cfg,
                'metrics': metrics,
            }

        except Exception as exc:
            return {
                'success': False,
                'error': str(exc),
                'output_path': output_path,
            }
