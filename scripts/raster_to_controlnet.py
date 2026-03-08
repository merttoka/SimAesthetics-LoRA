"""Convert satellite/scientific raster imagery to ControlNet-ready PNGs.

Bridges ORNL DAAC ecological data (vegetation change maps, biomass density
grids, GeoTIFF) into the ComfyUI pipeline. Lightweight: only requires Pillow.

Usage:
    python raster_to_controlnet.py --input biomass_density.tif --output controlnet_input.png --mode grayscale --size 1024
    python raster_to_controlnet.py --input satellite_data/ --output controlnet_inputs/ --mode binary --threshold 0.5
    python raster_to_controlnet.py --input vegetation_change.tif --output depth_map.png --mode grayscale --invert --blur 3
"""

import argparse
import struct
from pathlib import Path

from PIL import Image, ImageFilter

# Standard image extensions Pillow handles natively
STANDARD_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".gif"}
RASTER_EXTS = {".tif", ".tiff", ".geotiff"}


def viridis_lut():
    """Generate a viridis-like green gradient LUT (256 entries, RGB)."""
    lut = []
    for i in range(256):
        t = i / 255.0
        # Simplified viridis: dark purple -> teal -> yellow-green
        r = int(255 * max(0, min(1, -0.8 + 3.0 * t * t)))
        g = int(255 * max(0, min(1, 0.1 + 0.9 * t - 0.3 * t * t)))
        b = int(255 * max(0, min(1, 0.5 + 0.5 * t - 1.5 * t * t)))
        lut.extend([r, g, b])
    return lut


def load_raster(path: Path) -> Image.Image:
    """Load raster data, handling GeoTIFF float data gracefully.

    Tries Pillow first. If that fails on multi-band/float TIFFs, attempts
    raw numpy parsing. Prints upgrade suggestion if all else fails.
    """
    # Standard image — just open directly
    if path.suffix.lower() in STANDARD_EXTS:
        return Image.open(path).convert("F")

    # Try Pillow's TIFF reader
    try:
        img = Image.open(path)
        # Convert to float for uniform normalization
        if img.mode == "F":
            return img
        if img.mode in ("I", "I;16", "I;16B"):
            return img.convert("F")
        if img.mode in ("RGB", "RGBA"):
            return img.convert("L").convert("F")
        return img.convert("F")
    except Exception as pil_err:
        print(f"Pillow couldn't read {path.name}: {pil_err}")

    # Try numpy fallback for raw float32 rasters
    try:
        import numpy as np
        raw = np.fromfile(str(path), dtype=np.float32)
        # Guess square dimensions
        side = int(raw.size ** 0.5)
        if side * side != raw.size:
            print(f"Non-square raw data ({raw.size} values), trimming to {side}x{side}")
            raw = raw[:side * side]
        arr = raw.reshape(side, side)
        # Handle NaN/nodata
        valid = arr[~np.isnan(arr)]
        if valid.size == 0:
            raise ValueError("All NaN values")
        arr = np.nan_to_num(arr, nan=float(np.nanmin(arr)))
        return Image.fromarray(arr, mode="F")
    except ImportError:
        pass
    except Exception as np_err:
        print(f"numpy fallback failed: {np_err}")

    print(
        f"\nCouldn't read {path.name}. For full GeoTIFF/NetCDF support:\n"
        f"  pip install rasterio\n"
        f"Then use: rasterio.open('{path}').read(1)\n"
    )
    raise SystemExit(1)


def normalize_to_uint8(img: Image.Image) -> Image.Image:
    """Normalize float image to 0-255 range."""
    # Get min/max
    extrema = img.getextrema()
    lo, hi = float(extrema[0]), float(extrema[1])

    if hi - lo < 1e-10:
        # Constant image
        return Image.new("L", img.size, 128)

    # Linear normalize: pixel = (pixel - lo) / (hi - lo) * 255
    offset = -lo
    scale = 255.0 / (hi - lo)
    normalized = img.point(lambda p: (p + offset) * scale)
    return normalized.convert("L")


def apply_colormap(img_l: Image.Image, mode: str, threshold: float = 0.5) -> Image.Image:
    """Apply colormap to grayscale image.

    Args:
        img_l: 8-bit grayscale image
        mode: 'grayscale', 'viridis', or 'binary'
        threshold: threshold for binary mode (0-1, applied to 0-255 range)
    """
    if mode == "grayscale":
        return img_l

    if mode == "binary":
        thresh_val = int(threshold * 255)
        return img_l.point(lambda p: 255 if p >= thresh_val else 0)

    if mode == "viridis":
        rgb = img_l.convert("RGB")
        lut = viridis_lut()
        return rgb.point(lut)

    raise ValueError(f"Unknown mode: {mode}. Use grayscale, viridis, or binary.")


def process_raster(
    input_path: Path,
    output_path: Path,
    mode: str = "grayscale",
    size: int = 1024,
    invert: bool = False,
    blur: float = 0,
    threshold: float = 0.5,
):
    """Process a single raster file to ControlNet-ready PNG."""
    print(f"Processing: {input_path.name}")

    # Load and normalize
    img_f = load_raster(input_path)
    img_l = normalize_to_uint8(img_f)

    # Invert (for depth maps where high=close)
    if invert:
        img_l = img_l.point(lambda p: 255 - p)

    # Apply colormap
    result = apply_colormap(img_l, mode, threshold)

    # Resize
    if result.size != (size, size):
        result = result.resize((size, size), Image.LANCZOS)

    # Gaussian blur for smoothing satellite pixel artifacts
    if blur > 0:
        result = result.filter(ImageFilter.GaussianBlur(radius=blur))

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.save(output_path, "PNG")
    print(f"  -> {output_path} ({result.size[0]}x{result.size[1]}, {mode})")


def main():
    parser = argparse.ArgumentParser(
        description="Convert satellite/scientific rasters to ControlNet-ready PNGs"
    )
    parser.add_argument("--input", "-i", required=True, help="Input raster or directory")
    parser.add_argument("--output", "-o", required=True, help="Output PNG or directory")
    parser.add_argument(
        "--mode", "-m", default="grayscale",
        choices=["grayscale", "viridis", "binary"],
        help="Colormap mode (default: grayscale)",
    )
    parser.add_argument("--size", "-s", type=int, default=1024, help="Output resolution (default: 1024)")
    parser.add_argument("--invert", action="store_true", help="Invert values (for depth maps)")
    parser.add_argument("--blur", type=float, default=0, help="Gaussian blur radius")
    parser.add_argument("--threshold", type=float, default=0.5, help="Binary threshold 0-1 (default: 0.5)")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if input_path.is_dir():
        # Batch mode
        output_path.mkdir(parents=True, exist_ok=True)
        exts = STANDARD_EXTS | RASTER_EXTS
        files = sorted(f for f in input_path.iterdir() if f.suffix.lower() in exts)

        if not files:
            print(f"No raster/image files found in {input_path}")
            return

        print(f"Batch: {len(files)} files")
        for f in files:
            out = output_path / f"{f.stem}.png"
            try:
                process_raster(input_path=f, output_path=out,
                               mode=args.mode, size=args.size, invert=args.invert,
                               blur=args.blur, threshold=args.threshold)
            except Exception as e:
                print(f"  SKIP {f.name}: {e}")
    else:
        process_raster(
            input_path, output_path,
            mode=args.mode, size=args.size, invert=args.invert,
            blur=args.blur, threshold=args.threshold,
        )


if __name__ == "__main__":
    main()
