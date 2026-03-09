"""Composite AI-rendered crops back onto original ultrawide sim frames.

Uses manifest.json from prepare_dataset.py to map crops back to their
original positions. Produces overlay frames for video export.

Usage:
    # Basic composite (middle crop)
    python overlay_composite.py \
        --manifest datasets/sim_aesthetic/manifest.json \
        --source-dir ../recordings/ \
        --ai-dir outputs/img2img_lora/ \
        --out overlay_frames/

    # Side-by-side (composite top, original bottom, cropped to middle 40%)
    python overlay_composite.py \
        --manifest datasets/sim_aesthetic/manifest.json \
        --source-dir ../recordings/ \
        --ai-dir outputs/img2img_lora/ \
        --out overlay_frames/ \
        --side-by-side --h-crop 0.4

    # Variations mode: scattered small AI patches instead of full 1024² blocks
    python overlay_composite.py \
        --manifest datasets/sim_aesthetic/manifest.json \
        --source-dir ../recordings/ \
        --ai-dir outputs/img2img_lora/ \
        --out overlay_frames/ \
        --variations 6 --patch-range 192,512

    # Then ffmpeg to video:
    # ffmpeg -framerate 12 -i overlay_frames/frame_%04d.png -c:v libx264 -pix_fmt yuv420p overlay.mp4
"""

import argparse
import json
import random
from pathlib import Path
from collections import defaultdict
from PIL import Image, ImageDraw


def load_manifest(manifest_path: Path) -> list[dict]:
    data = json.loads(manifest_path.read_text())
    return data["entries"]


def group_by_source(entries: list[dict]) -> dict[str, list[dict]]:
    """Group manifest entries by source frame filename."""
    groups = defaultdict(list)
    for entry in entries:
        groups[entry["source"]].append(entry)
    return dict(groups)


def crop_horizontal_center(img: Image.Image, h_crop: float) -> Image.Image:
    """Crop to the middle h_crop fraction of horizontal space."""
    w, h = img.size
    crop_w = int(w * h_crop)
    x0 = (w - crop_w) // 2
    return img.crop((x0, 0, x0 + crop_w, h))


def sample_non_overlapping_patches(
    crop_size: int,
    n_patches: int,
    min_patch: int,
    max_patch: int,
    rng: random.Random,
) -> list[tuple[int, int, int]]:
    """Sample non-overlapping square patches within a crop_size region.

    Returns list of (x, y, size) relative to the crop origin.
    """
    patches = []
    attempts = 0
    max_attempts = n_patches * 50

    while len(patches) < n_patches and attempts < max_attempts:
        attempts += 1
        size = rng.randint(min_patch, max_patch)
        x = rng.randint(0, crop_size - size)
        y = rng.randint(0, crop_size - size)

        # Check overlap with existing patches
        overlaps = False
        for px, py, ps in patches:
            if x < px + ps and x + size > px and y < py + ps and y + size > py:
                overlaps = True
                break

        if not overlaps:
            patches.append((x, y, size))

    return patches


def composite_frame(
    source_img: Image.Image,
    crops: list[dict],
    ai_dir: Path,
    opacity: float = 0.85,
    border: int = 2,
    border_color: tuple = (0, 255, 255),
) -> Image.Image:
    """Paste AI crops back onto source frame at original coordinates."""
    result = source_img.copy()
    draw = ImageDraw.Draw(result)

    for entry in crops:
        ai_path = ai_dir / f"ai_img_{entry['index']}.png"
        if not ai_path.exists():
            continue

        ai_img = Image.open(ai_path).convert("RGB")
        crop_size = entry["crop_size"]
        x, y = entry["x"], entry["y"]

        # Resize AI output back to original crop dimensions
        ai_resized = ai_img.resize((crop_size, crop_size), Image.LANCZOS)

        # Blend with opacity
        region = source_img.crop((x, y, x + crop_size, y + crop_size))
        blended = Image.blend(region, ai_resized, opacity)
        result.paste(blended, (x, y))

        # Draw border
        if border > 0:
            draw.rectangle(
                [x, y, x + crop_size - 1, y + crop_size - 1],
                outline=border_color, width=border,
            )

    return result


def composite_variations(
    source_img: Image.Image,
    crops: list[dict],
    ai_dir: Path,
    n_patches: int = 6,
    min_patch: int = 192,
    max_patch: int = 512,
    opacity: float = 0.9,
    border: int = 1,
    border_color: tuple = (0, 255, 255),
    seed: int = 42,
) -> Image.Image:
    """Scatter small AI patches onto source frame instead of full crops."""
    result = source_img.copy()
    draw = ImageDraw.Draw(result)
    rng = random.Random(seed)

    for entry in crops:
        ai_path = ai_dir / f"ai_img_{entry['index']}.png"
        if not ai_path.exists():
            continue

        ai_img = Image.open(ai_path).convert("RGB")
        crop_size = entry["crop_size"]
        origin_x, origin_y = entry["x"], entry["y"]

        # Resize AI output to match original crop dimensions
        ai_resized = ai_img.resize((crop_size, crop_size), Image.LANCZOS)

        # Scale patch sizes relative to crop_size (manifest crops may not be 1024)
        scale = crop_size / 1024.0
        scaled_min = max(int(min_patch * scale), 32)
        scaled_max = min(int(max_patch * scale), crop_size)

        patches = sample_non_overlapping_patches(
            crop_size, n_patches, scaled_min, scaled_max, rng
        )

        for px, py, ps in patches:
            # Extract AI patch
            ai_patch = ai_resized.crop((px, py, px + ps, py + ps))

            # Corresponding region in source
            sx, sy = origin_x + px, origin_y + py
            src_patch = source_img.crop((sx, sy, sx + ps, sy + ps))

            # Blend and paste
            blended = Image.blend(src_patch, ai_patch, opacity)
            result.paste(blended, (sx, sy))

            if border > 0:
                draw.rectangle(
                    [sx, sy, sx + ps - 1, sy + ps - 1],
                    outline=border_color, width=border,
                )

    return result


def make_side_by_side(original: Image.Image, overlay: Image.Image, gap: int = 4) -> Image.Image:
    """Stack overlay on top, original below."""
    w, h = original.size
    combined = Image.new("RGB", (w, h * 2 + gap), (0, 0, 0))
    combined.paste(overlay, (0, 0))
    combined.paste(original, (0, h + gap))
    return combined


def main():
    parser = argparse.ArgumentParser(description="Composite AI crops onto source frames")
    parser.add_argument("--manifest", "-m", required=True, help="manifest.json from prepare_dataset.py")
    parser.add_argument("--source-dir", "-s", required=True, help="Original ultrawide frames directory")
    parser.add_argument("--ai-dir", "-a", required=True, help="AI-rendered outputs directory")
    parser.add_argument("--out", "-o", default="overlay_frames", help="Output directory")
    parser.add_argument("--opacity", type=float, default=0.85, help="AI overlay opacity (0-1)")
    parser.add_argument("--border", type=int, default=2, help="Border width around crops (0=none)")
    parser.add_argument("--side-by-side", action="store_true", help="Composite top, original bottom")
    parser.add_argument("--scale", type=float, default=0.5, help="Output scale factor (1.0=full res)")
    parser.add_argument("--h-crop", type=float, default=0.4, help="Horizontal crop fraction (0.4 = middle 40%%)")
    parser.add_argument("--variations", type=int, default=0, help="Variations mode: N non-overlapping patches per crop (e.g. 6)")
    parser.add_argument("--patch-range", default="192,512", help="Min,max patch size for variations mode (e.g. 192,512)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for variations placement")
    args = parser.parse_args()

    min_patch, max_patch = [int(x) for x in args.patch_range.split(",")]

    manifest_path = Path(args.manifest)
    source_dir = Path(args.source_dir)
    ai_dir = Path(args.ai_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    entries = load_manifest(manifest_path)
    groups = group_by_source(entries)

    mode = "variations" if args.variations > 0 else "full"
    print(f"Manifest: {len(entries)} crops from {len(groups)} source frames")
    print(f"AI dir: {ai_dir}")
    print(f"Mode: {mode}" + (f" ({args.variations} patches, {min_patch}-{max_patch}px)" if mode == "variations" else ""))

    for fi, (source_name, crops) in enumerate(sorted(groups.items())):
        source_path = source_dir / source_name
        if not source_path.exists():
            print(f"  SKIP {source_name} (not found)")
            continue

        source_img = Image.open(source_path).convert("RGB")

        if args.variations > 0:
            overlay = composite_variations(
                source_img, crops, ai_dir,
                n_patches=args.variations,
                min_patch=min_patch, max_patch=max_patch,
                opacity=args.opacity, border=args.border,
                seed=args.seed + fi,
            )
        else:
            overlay = composite_frame(source_img, crops, ai_dir, args.opacity, args.border)

        if args.side_by_side:
            # Crop both to horizontal center before stacking
            overlay_cropped = crop_horizontal_center(overlay, args.h_crop)
            original_cropped = crop_horizontal_center(source_img, args.h_crop)
            output = make_side_by_side(original_cropped, overlay_cropped)
        else:
            output = crop_horizontal_center(overlay, args.h_crop)

        # Scale down for manageable file sizes
        if args.scale != 1.0:
            new_size = (int(output.size[0] * args.scale), int(output.size[1] * args.scale))
            output = output.resize(new_size, Image.LANCZOS)

        # Extract frame number from source filename (e.g. "Image Sequence_002_0500.jpg" → "0500")
        stem = Path(source_name).stem
        frame_num = stem.split("_")[-1] if "_" in stem else f"{fi+1:03d}"
        out_path = out_dir / f"frame_{frame_num}.png"
        output.save(out_path, "PNG")
        print(f"  [{fi+1}/{len(groups)}] {source_name} → {out_path.name} ({len(crops)} crops)")

    print(f"\nDone: {len(groups)} frames in {out_dir}/")
    print(f"To make video: ffmpeg -framerate 12 -i {out_dir}/frame_%04d.png -c:v libx264 -pix_fmt yuv420p overlay.mp4")


if __name__ == "__main__":
    main()
