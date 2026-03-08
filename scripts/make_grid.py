"""Build side-by-side comparison grids from sim frames and AI outputs.

Usage:
    # Auto-pair by filename (input img_080.png → output ai_img_080.png)
    python make_grid.py --input ../datasets/sim_aesthetic/ --output ../outputs/ --out grid.png

    # Manual pairs (comma-separated input:output)
    python make_grid.py --pairs "img_080.png:controlnet_00017.png,img_150.png:controlnet_00020.png" --out grid.png

    # From two directories, matched by sort order
    python make_grid.py --left ../datasets/sim_aesthetic/ --right ../outputs/ --out grid.png --limit 6
"""

import argparse
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


def load_and_resize(path: Path, size: int) -> Image.Image:
    return Image.open(path).convert("RGB").resize((size, size), Image.LANCZOS)


def make_grid(
    pairs: list[tuple[Path, Path]],
    cell_size: int = 512,
    gap: int = 8,
    margin: int = 16,
    bg_color: tuple = (0, 0, 0),
    label_left: str = "Simulation",
    label_right: str = "AI Rendered",
) -> Image.Image:
    """Build a grid with left=input, right=output, one pair per row."""
    n = len(pairs)
    col_w = cell_size
    row_h = cell_size
    header_h = 40

    total_w = margin + col_w + gap + col_w + margin
    total_h = margin + header_h + (row_h + gap) * n - gap + margin

    grid = Image.new("RGB", (total_w, total_h), bg_color)
    draw = ImageDraw.Draw(grid)

    # Try to load a font, fall back to default
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 24)
    except (OSError, IOError):
        try:
            font = ImageFont.truetype("arial.ttf", 24)
        except (OSError, IOError):
            font = ImageFont.load_default()

    # Headers
    left_x = margin + col_w // 2
    right_x = margin + col_w + gap + col_w // 2
    draw.text((left_x, margin + 4), label_left, fill=(180, 180, 180), font=font, anchor="mt")
    draw.text((right_x, margin + 4), label_right, fill=(180, 180, 180), font=font, anchor="mt")

    # Rows
    for i, (left_path, right_path) in enumerate(pairs):
        y = margin + header_h + i * (row_h + gap)

        left_img = load_and_resize(left_path, cell_size)
        grid.paste(left_img, (margin, y))

        right_img = load_and_resize(right_path, cell_size)
        grid.paste(right_img, (margin + col_w + gap, y))

    return grid


def find_pairs_by_order(left_dir: Path, right_dir: Path, limit: int = 6) -> list[tuple[Path, Path]]:
    """Match files from two directories by sort order."""
    exts = {".png", ".jpg", ".jpeg", ".webp"}
    lefts = sorted(f for f in left_dir.iterdir() if f.suffix.lower() in exts)
    rights = sorted(f for f in right_dir.iterdir() if f.suffix.lower() in exts)
    n = min(len(lefts), len(rights), limit)
    return list(zip(lefts[:n], rights[:n]))


def parse_manual_pairs(pairs_str: str, base: Path = Path(".")) -> list[tuple[Path, Path]]:
    """Parse 'left:right,left:right' string into path pairs."""
    pairs = []
    for pair in pairs_str.split(","):
        left, right = pair.strip().split(":")
        pairs.append((base / left.strip(), base / right.strip()))
    return pairs


def main():
    parser = argparse.ArgumentParser(description="Build side-by-side comparison grid")
    parser.add_argument("--left", "-l", help="Left (input) image directory")
    parser.add_argument("--right", "-r", help="Right (output) image directory")
    parser.add_argument("--pairs", "-p", help="Manual pairs: 'left:right,left:right'")
    parser.add_argument("--out", "-o", default="grid.png", help="Output grid image path")
    parser.add_argument("--limit", type=int, default=6, help="Max rows (default: 6)")
    parser.add_argument("--size", type=int, default=512, help="Cell size in pixels (default: 512)")
    parser.add_argument("--label-left", default="Simulation", help="Left column label")
    parser.add_argument("--label-right", default="AI Rendered", help="Right column label")
    args = parser.parse_args()

    if args.pairs:
        pairs = parse_manual_pairs(args.pairs)
    elif args.left and args.right:
        pairs = find_pairs_by_order(Path(args.left), Path(args.right), args.limit)
    else:
        parser.error("Provide --pairs or --left + --right")

    if not pairs:
        print("No pairs found")
        return

    print(f"Building grid: {len(pairs)} rows, {args.size}px cells")
    grid = make_grid(
        pairs, cell_size=args.size,
        label_left=args.label_left, label_right=args.label_right,
    )
    grid.save(args.out, "PNG")
    print(f"Saved: {args.out} ({grid.size[0]}x{grid.size[1]})")


if __name__ == "__main__":
    main()
