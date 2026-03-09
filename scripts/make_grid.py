"""Build comparison grids from sim frames and AI outputs.

Layout: columns extend horizontally, each column is a pair stacked vertically
(sim on top, AI below). Row labels on the left, frame numbers underneath.

Usage:
    # From two directories, matched by sort order
    python make_grid.py --left ../datasets/sim_aesthetic/ --right ../outputs/ --out grid.png

    # Pick a range of images (0-indexed)
    python make_grid.py --left ... --right ... --start 10 --end 20 --out grid.png

    # Timelapse: one crop per source frame, sorted by frame number
    python make_grid.py --left ... --right ... --timelapse --manifest datasets/sim_aesthetic/manifest.json --out grid.png

    # Timelapse with batching
    python make_grid.py --left ... --right ... --timelapse --manifest ... --count 8 --iter 3 --out grid.png
"""

import argparse
import json
import math
import re
from collections import OrderedDict
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


def load_and_resize(path: Path, size: int) -> Image.Image:
    return Image.open(path).convert("RGB").resize((size, size), Image.LANCZOS)


def get_font(size: int = 20):
    for name in ["/System/Library/Fonts/Helvetica.ttc", "arial.ttf"]:
        try:
            return ImageFont.truetype(name, size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


def extract_frame_number(source_name: str) -> int:
    """Extract frame number from source filename like 'Image Sequence_002_0500.jpg'."""
    nums = re.findall(r'\d+', source_name)
    return int(nums[-1]) if nums else 0


def best_grid_layout(n: int, max_cols: int = 8) -> tuple[int, int]:
    """Find cols x rows where cols >= rows (landscape orientation).

    Each pair is 2 cells tall (sim + AI stacked), so we optimize for
    a wide layout. Always prefer more columns than rows.
    """
    if n <= max_cols:
        return n, 1
    best = (max_cols, math.ceil(n / max_cols))
    for cols in range(max_cols, 1, -1):
        rows = math.ceil(n / cols)
        if cols >= rows:
            return cols, rows
    return best


def make_grid(
    pairs: list[tuple[Path, Path]],
    labels: list[str] | None = None,
    cell_size: int = 512,
    gap: int = 8,
    margin: int = 16,
    bg_color: tuple = (0, 0, 0),
    label_top: str = "Simulation",
    label_bottom: str = "AI Rendered",
    max_cols: int = 8,
) -> Image.Image:
    """Build a grid: each pair is stacked vertically (sim on top, AI below).

    Auto-wraps into multiple rows when pairs > max_cols, targeting ~16:9 aspect.
    Labels on the left of each row pair, frame labels underneath each column.
    """
    n = len(pairs)
    cols, rows = best_grid_layout(n, max_cols)
    label_w = 160
    footer_h = 24
    pair_h = cell_size + gap + cell_size + footer_h  # sim + AI + label
    row_gap = gap * 2

    total_w = margin + label_w + (cell_size + gap) * cols - gap + margin
    total_h = margin + (pair_h + row_gap) * rows - row_gap + margin

    grid = Image.new("RGB", (total_w, total_h), bg_color)
    draw = ImageDraw.Draw(grid)

    font = get_font(20)
    font_small = get_font(14)

    for idx, (top_path, bot_path) in enumerate(pairs):
        row = idx // cols
        col = idx % cols
        x = margin + label_w + col * (cell_size + gap)
        y = margin + row * (pair_h + row_gap)

        # Row labels on left (only for first column)
        if col == 0:
            draw.text((margin + label_w // 2, y + cell_size // 2),
                       label_top, fill=(180, 180, 180), font=font, anchor="mm")
            draw.text((margin + label_w // 2, y + cell_size + gap + cell_size // 2),
                       label_bottom, fill=(180, 180, 180), font=font, anchor="mm")

        top_img = load_and_resize(top_path, cell_size)
        grid.paste(top_img, (x, y))

        bot_img = load_and_resize(bot_path, cell_size)
        grid.paste(bot_img, (x, y + cell_size + gap))

        # Frame label underneath
        frame_label = labels[idx] if labels else top_path.stem
        draw.text(
            (x + cell_size // 2, y + cell_size + gap + cell_size + 4),
            frame_label, fill=(120, 120, 120), font=font_small, anchor="mt",
        )

    return grid


def find_pairs_by_order(
    left_dir: Path, right_dir: Path, start: int = 0, end: int | None = None,
) -> list[tuple[Path, Path]]:
    """Match files from two directories by sort order, with start/end slicing."""
    exts = {".png", ".jpg", ".jpeg", ".webp"}
    lefts = sorted(f for f in left_dir.iterdir() if f.suffix.lower() in exts)
    rights = sorted(f for f in right_dir.iterdir() if f.suffix.lower() in exts)
    n = min(len(lefts), len(rights))
    if end is None:
        end = n
    end = min(end, n)
    start = max(0, min(start, end))
    return list(zip(lefts[start:end], rights[start:end]))


def find_timelapse_pairs(
    manifest_path: Path, left_dir: Path, right_dir: Path,
) -> tuple[list[tuple[Path, Path]], list[str]]:
    """Pick one crop per source frame, sorted by frame number.

    Returns (pairs, labels) where labels are frame numbers like 'f0500'.
    """
    data = json.loads(manifest_path.read_text())
    entries = data["entries"]

    # Group by source, keep first crop per source
    seen = OrderedDict()
    for entry in entries:
        src = entry["source"]
        if src not in seen:
            seen[src] = entry

    # Sort by frame number extracted from source filename
    sorted_entries = sorted(seen.values(), key=lambda e: extract_frame_number(e["source"]))

    pairs = []
    labels = []
    for entry in sorted_entries:
        idx = entry["index"]
        left = left_dir / f"img_{idx}.png"
        right = right_dir / f"ai_img_{idx}.png"
        if left.exists() and right.exists():
            frame_num = extract_frame_number(entry["source"])
            pairs.append((left, right))
            labels.append(f"f{frame_num}")

    return pairs, labels


def parse_manual_pairs(pairs_str: str, base: Path = Path(".")) -> list[tuple[Path, Path]]:
    """Parse 'left:right,left:right' string into path pairs."""
    pairs = []
    for pair in pairs_str.split(","):
        left, right = pair.strip().split(":")
        pairs.append((base / left.strip(), base / right.strip()))
    return pairs


def main():
    parser = argparse.ArgumentParser(description="Build comparison grid")
    parser.add_argument("--left", "-l", help="Top row (input) image directory")
    parser.add_argument("--right", "-r", help="Bottom row (output) image directory")
    parser.add_argument("--pairs", "-p", help="Manual pairs: 'left:right,left:right'")
    parser.add_argument("--manifest", "-m", help="manifest.json for --timelapse mode")
    parser.add_argument("--timelapse", action="store_true", help="One crop per source frame, sorted by frame number")
    parser.add_argument("--out", "-o", default="grid.png", help="Output grid image path")
    parser.add_argument("--start", type=int, default=0, help="Start index (0-indexed)")
    parser.add_argument("--end", type=int, default=None, help="End index (exclusive, default: all)")
    parser.add_argument("--count", type=int, default=None, help="Pairs per grid (use with --iter)")
    parser.add_argument("--iter", type=int, default=1, help="Number of grids to generate (use with --count)")
    parser.add_argument("--size", type=int, default=512, help="Cell size in pixels (default: 512)")
    parser.add_argument("--max-cols", type=int, default=8, help="Max columns before wrapping (default: 8)")
    parser.add_argument("--label-top", default="Simulation", help="Top row label")
    parser.add_argument("--label-bottom", default="AI Rendered", help="Bottom row label")
    args = parser.parse_args()

    labels = None

    if args.timelapse:
        if not args.manifest or not args.left or not args.right:
            parser.error("--timelapse requires --manifest, --left, and --right")
        all_pairs, labels = find_timelapse_pairs(
            Path(args.manifest), Path(args.left), Path(args.right),
        )
    elif args.pairs:
        all_pairs = parse_manual_pairs(args.pairs)
    elif args.left and args.right:
        all_pairs = find_pairs_by_order(Path(args.left), Path(args.right), args.start, args.end)
    else:
        parser.error("Provide --pairs, --left + --right, or --timelapse + --manifest")

    if not all_pairs:
        print("No pairs found")
        return

    # Apply start/end slicing
    end = args.end if args.end is not None else len(all_pairs)
    end = min(end, len(all_pairs))
    start = max(0, min(args.start, end))
    all_pairs = all_pairs[start:end]
    if labels:
        labels = labels[start:end]

    # Split into batches if --count is set
    if args.count:
        batches = []
        for i in range(args.iter):
            s = i * args.count
            e = s + args.count
            batch = all_pairs[s:e]
            batch_labels = labels[s:e] if labels else None
            if batch:
                batches.append((start + s, batch, batch_labels))
    else:
        batches = [(start, all_pairs, labels)]

    out_base = Path(args.out)
    for batch_start, pairs, batch_labels in batches:
        end_idx = batch_start + len(pairs)
        print(f"Building grid: {len(pairs)} columns, {args.size}px cells [{batch_start}-{end_idx}]")
        grid = make_grid(
            pairs, labels=batch_labels, cell_size=args.size,
            label_top=args.label_top, label_bottom=args.label_bottom,
            max_cols=args.max_cols,
        )
        out_path = out_base.with_stem(f"{out_base.stem}_{batch_start}-{end_idx}")
        grid.save(out_path, "PNG")
        print(f"Saved: {out_path} ({grid.size[0]}x{grid.size[1]})")


if __name__ == "__main__":
    main()
