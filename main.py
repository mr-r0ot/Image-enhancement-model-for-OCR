"""
Persian Handwriting Image Enhancer
====================================
Preprocess low-quality photos of handwritten Persian text so they are
ready for the OCR stage.

Usage
-----
  # Fully automatic (AI detects everything):
  python main.py photo.jpg enhanced.png

  # With explicit output directory:
  python main.py photo.jpg enhanced.png --output-dir ./results

  # Override specific settings:
  python main.py photo.jpg enhanced.png --binarize sauvola --denoise nlm

  # Disable a step:
  python main.py photo.jpg enhanced.png --no-deskew --no-bg-norm

  # Load all settings from a JSON file:
  python main.py photo.jpg enhanced.png --config my_settings.json

  # Verbose mode (prints each processing step):
  python main.py photo.jpg enhanced.png -v

  # Analyse image without processing:
  python main.py photo.jpg --analyse-only

JSON config example (all keys are optional):
  {
    "deskew": true,
    "normalize_background": true,
    "clahe_clip_limit": 3.0,
    "denoise_method": "nlm",
    "nlm_h": 12.0,
    "binarize_method": "sauvola",
    "sauvola_window": 31,
    "sauvola_k": 0.2,
    "morph_cleanup": true,
    "min_component_area": 20,
    "upscale": true,
    "target_min_dimension": 1500,
    "invert_output": false
  }
"""

import argparse
import json
import sys
from pathlib import Path

from auto_config import auto_configure
from enhancer import ImageEnhancer
import cv2


# -- argument parsing ----------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog='python main.py',
        description='Enhance low-quality Persian handwriting photos for OCR.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    p.add_argument('input',
                   help='Path to the input image (jpg/png/bmp/tiff/...).')
    p.add_argument('output', nargs='?', default=None,
                   help='Output filename (e.g. result.png). '
                        'Required unless --analyse-only is used.')

    p.add_argument('--output-dir', default=None, metavar='DIR',
                   help='Directory to write the output file into. '
                        'Default: same folder as the input image.')

    # -- analyse-only ----------------------------------------------------------
    p.add_argument('--analyse-only', action='store_true',
                   help='Print detected image metrics and recommended settings, '
                        'then exit without writing any file.')

    # -- manual overrides (all optional) --------------------------------------
    g = p.add_argument_group(
        'Manual overrides',
        'If not given, every parameter is auto-detected from the image.',
    )

    # deskew
    deskew_g = g.add_mutually_exclusive_group()
    deskew_g.add_argument('--deskew', action='store_true', default=None,
                           help='Force enable deskew.')
    deskew_g.add_argument('--no-deskew', action='store_false', dest='deskew',
                           help='Disable deskew.')

    # background normalisation
    bgnorm_g = g.add_mutually_exclusive_group()
    bgnorm_g.add_argument('--bg-norm', action='store_true', default=None,
                           dest='normalize_background',
                           help='Force enable background normalisation.')
    bgnorm_g.add_argument('--no-bg-norm', action='store_false',
                           dest='normalize_background',
                           help='Disable background normalisation.')

    # CLAHE
    clahe_g = g.add_mutually_exclusive_group()
    clahe_g.add_argument('--clahe', action='store_true', default=None,
                          help='Force enable CLAHE.')
    clahe_g.add_argument('--no-clahe', action='store_false', dest='clahe',
                          help='Disable CLAHE.')

    g.add_argument('--clahe-clip', type=float, metavar='FLOAT',
                   dest='clahe_clip_limit',
                   help='CLAHE clip limit (default auto, usually 2.0-3.5).')

    g.add_argument('--denoise',
                   choices=['nlm', 'bilateral', 'gaussian', 'none'],
                   dest='denoise_method',
                   help='Denoising algorithm.')

    g.add_argument('--binarize',
                   choices=['sauvola', 'otsu', 'adaptive', 'none'],
                   dest='binarize_method',
                   help='Binarization algorithm.')

    g.add_argument('--sauvola-window', type=int, metavar='INT',
                   dest='sauvola_window',
                   help='Sauvola window size in pixels (odd, default auto).')

    g.add_argument('--sauvola-k', type=float, metavar='FLOAT',
                   dest='sauvola_k',
                   help='Sauvola k parameter (default 0.2).')

    g.add_argument('--scale', type=float, metavar='FLOAT',
                   dest='max_upscale_factor',
                   help='Maximum upscale factor (e.g. 2.0).')

    g.add_argument('--no-cleanup', action='store_false', dest='morph_cleanup',
                   default=None,
                   help='Disable morphological cleanup.')

    g.add_argument('--invert', action='store_true', dest='invert_output',
                   default=None,
                   help='Invert output (white text on black background).')

    # JSON config
    p.add_argument('--config', metavar='JSON',
                   help='Path to a JSON file with config overrides '
                        '(merged with CLI flags; CLI takes precedence).')

    p.add_argument('-v', '--verbose', action='store_true',
                   help='Print processing details to stdout.')

    return p


def collect_overrides(args) -> dict:
    """Collect only the CLI flags that were explicitly set by the user."""
    overrides = {}
    mappings = [
        ('deskew', 'deskew'),
        ('normalize_background', 'normalize_background'),
        ('clahe', 'clahe'),
        ('clahe_clip_limit', 'clahe_clip_limit'),
        ('denoise_method', 'denoise_method'),
        ('binarize_method', 'binarize_method'),
        ('sauvola_window', 'sauvola_window'),
        ('sauvola_k', 'sauvola_k'),
        ('max_upscale_factor', 'max_upscale_factor'),
        ('morph_cleanup', 'morph_cleanup'),
        ('invert_output', 'invert_output'),
    ]
    for attr, key in mappings:
        val = getattr(args, attr, None)
        if val is not None:
            overrides[key] = val
    return overrides


# -- analyse-only helper -------------------------------------------------------

def run_analyse_only(image_path: str):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: cannot read image: {image_path}")
        sys.exit(1)

    cfg, metrics = auto_configure(img)

    print("\n=== Image Metrics ===")
    print(f"  Resolution      : {img.shape[1]}x{img.shape[0]} px")
    print(f"  Noise level     : {metrics['noise_level']:.1f}  "
          f"({'high' if metrics['noise_level'] > 400 else 'medium' if metrics['noise_level'] > 100 else 'low'})")
    print(f"  Lighting var    : {metrics['lighting_variation']:.3f}  "
          f"({'severe' if metrics['lighting_variation'] > 0.3 else 'moderate' if metrics['lighting_variation'] > 0.1 else 'good'})")
    print(f"  Contrast range  : {metrics['contrast_range']} / 255")
    print(f"  Skew angle      : {metrics['skew_angle']:+.1f}deg  "
          f"({'needs correction' if metrics['has_skew'] else 'OK'})")

    print("\n=== Recommended Config ===")
    print(f"  upscale            : {cfg.upscale}  (target {cfg.target_min_dimension}px, max {cfg.max_upscale_factor}x)")
    print(f"  deskew             : {cfg.deskew}  ({cfg.deskew_method})")
    print(f"  normalize_bg       : {cfg.normalize_background}")
    print(f"  clahe              : {cfg.clahe}  (clip={cfg.clahe_clip_limit})")
    print(f"  denoise            : {cfg.denoise}  ({cfg.denoise_method})")
    print(f"  binarize           : {cfg.binarize}  ({cfg.binarize_method})")
    print(f"  sauvola_window     : {cfg.sauvola_window}")
    print(f"  sauvola_k          : {cfg.sauvola_k}")
    print(f"  morph_cleanup      : {cfg.morph_cleanup}  (min_area={cfg.min_component_area})")


# -- main ----------------------------------------------------------------------

def main():
    parser = build_parser()
    args = parser.parse_args()

    # Validate input
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: input file not found: {args.input}")
        sys.exit(1)

    # Analyse-only mode
    if args.analyse_only:
        run_analyse_only(str(input_path))
        return

    # Need output name for normal mode
    if not args.output:
        parser.error("output filename is required (unless --analyse-only is used).")

    # Determine output path
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = input_path.parent

    output_path = output_dir / args.output

    # Build overrides: JSON file first, CLI flags on top (CLI wins)
    overrides = {}
    if args.config:
        cfg_path = Path(args.config)
        if not cfg_path.exists():
            print(f"Error: config file not found: {args.config}")
            sys.exit(1)
        with open(cfg_path, encoding='utf-8') as f:
            overrides.update(json.load(f))

    overrides.update(collect_overrides(args))

    # Run pipeline
    enhancer = ImageEnhancer(verbose=args.verbose)
    result = enhancer.process(
        input_path=str(input_path),
        output_path=str(output_path),
        user_overrides=overrides,
    )

    if result['success']:
        print(f"OK  ->  {result['output_path']}")
    else:
        print(f"Error: {result['error']}")
        sys.exit(1)


if __name__ == '__main__':
    main()
