"""Composite AI-rendered crops back onto original ultrawide sim frames.

Uses manifest.json from prepare_dataset.py to map crops back to their
original positions. Produces overlay frames for video export.

Usage:
    # Composite AI outputs onto source frames
    python overlay_composite.py \
        --manifest datasets/sim_aesthetic/manifest.json \
        --source-dir ../recordings/ \
        --ai-dir outputs/img2img_lora/ \
        --out overlay_frames/

    # Then ffmpeg to video:
    # ffmpeg -framerate 12 -i overlay_frames/frame_%03d.png -c:v libx264 -pix_fmt yuv420p overlay.mp4

    # Side-by-side video (original | overlay):
    python overlay_composite.py \
        --manifest datasets/sim_aesthetic/manifest.json \
        --source-dir ../recordings/ \
        --ai-dir outputs/img2img_lora/ \
        --out overlay_frames/ \
        --side-by-side
"""

import argparse
import json
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


def make_side_by_side(original: Image.Image, overlay: Image.Image, gap: int = 4) -> Image.Image:
    """Stack original and overlay side by side."""
    w, h = original.size
    combined = Image.new("RGB", (w * 2 + gap, h), (0, 0, 0))
    combined.paste(original, (0, 0))
    combined.paste(overlay, (w + gap, 0))
    return combined


def main():
    parser = argparse.ArgumentParser(description="Composite AI crops onto source frames")
    parser.add_argument("--manifest", "-m", required=True, help="manifest.json from prepare_dataset.py")
    parser.add_argument("--source-dir", "-s", required=True, help="Original ultrawide frames directory")
    parser.add_argument("--ai-dir", "-a", required=True, help="AI-rendered outputs directory")
    parser.add_argument("--out", "-o", default="overlay_frames", help="Output directory")
    parser.add_argument("--opacity", type=float, default=0.85, help="AI overlay opacity (0-1)")
    parser.add_argument("--border", type=int, default=2, help="Border width around crops (0=none)")
    parser.add_argument("--side-by-side", action="store_true", help="Output original|overlay side by side")
    parser.add_argument("--scale", type=float, default=0.5, help="Output scale factor (1.0=full res)")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    source_dir = Path(args.source_dir)
    ai_dir = Path(args.ai_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    entries = load_manifest(manifest_path)
    groups = group_by_source(entries)

    print(f"Manifest: {len(entries)} crops from {len(groups)} source frames")
    print(f"AI dir: {ai_dir}")

    for fi, (source_name, crops) in enumerate(sorted(groups.items())):
        source_path = source_dir / source_name
        if not source_path.exists():
            print(f"  SKIP {source_name} (not found)")
            continue

        source_img = Image.open(source_path).convert("RGB")
        overlay = composite_frame(source_img, crops, ai_dir, args.opacity, args.border)

        if args.side_by_side:
            output = make_side_by_side(source_img, overlay)
        else:
            output = overlay

        # Scale down for manageable file sizes
        if args.scale != 1.0:
            new_size = (int(output.size[0] * args.scale), int(output.size[1] * args.scale))
            output = output.resize(new_size, Image.LANCZOS)

        out_path = out_dir / f"frame_{fi+1:03d}.png"
        output.save(out_path, "PNG")
        print(f"  [{fi+1}/{len(groups)}] {source_name} → {out_path.name} ({len(crops)} crops)")

    print(f"\nDone: {len(groups)} frames in {out_dir}/")
    print(f"To make video: ffmpeg -framerate 12 -i {out_dir}/frame_%03d.png -c:v libx264 -pix_fmt yuv420p overlay.mp4")


if __name__ == "__main__":
    main()
