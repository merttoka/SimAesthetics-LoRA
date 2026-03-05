"""Sweep denoise/strength values for comparison grids.

Generates side-by-side comparisons across parameter ranges.
Useful for Day 2 (img2img denoise sweep) and Day 3 (ControlNet strength sweep).

Usage:
    python sweep_denoise.py --input frame.png --workflow ../workflows/sdxl_img2img.json \
        --param 3.denoise --values 0.3,0.5,0.7,0.9
    python sweep_denoise.py --input frame.png --workflow ../workflows/sdxl_controlnet_canny.json \
        --param 20.strength --values 0.4,0.6,0.8,1.0
"""

import argparse
import json
import shutil
from pathlib import Path

from comfyui_client import ComfyUIClient, load_workflow, parametrize_workflow

try:
    from PIL import Image, ImageDraw, ImageFont
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


def create_comparison_grid(
    images: list[tuple[str, Path]],
    output_path: Path,
    original: Path | None = None,
    cols: int = 4,
):
    """Create labeled comparison grid from sweep results."""
    if not HAS_PIL:
        print("Pillow not installed, skipping grid generation")
        return

    imgs = []
    if original:
        imgs.append(("original", Image.open(original)))
    for label, path in images:
        imgs.append((label, Image.open(path)))

    if not imgs:
        return

    w, h = imgs[0][1].size
    label_h = 40
    rows = (len(imgs) + cols - 1) // cols
    grid = Image.new("RGB", (w * cols, (h + label_h) * rows), (0, 0, 0))
    draw = ImageDraw.Draw(grid)

    for i, (label, img) in enumerate(imgs):
        col = i % cols
        row = i // cols
        x = col * w
        y = row * (h + label_h)
        grid.paste(img, (x, y + label_h))
        draw.text((x + 10, y + 10), label, fill=(255, 255, 255))

    grid.save(output_path)
    print(f"Grid saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Parameter sweep with comparison grid")
    parser.add_argument("--input", "-i", required=True, help="Input image")
    parser.add_argument("--workflow", "-w", required=True, help="Workflow JSON")
    parser.add_argument("--output", "-o", default="sweep_output", help="Output dir")
    parser.add_argument("--host", default="http://127.0.0.1:8188")
    parser.add_argument("--param", "-p", required=True, help="Parameter to sweep (node_id.field)")
    parser.add_argument("--values", "-v", required=True, help="Comma-separated values")
    parser.add_argument("--seed", type=int, default=42, help="Fixed seed for comparison")
    parser.add_argument("--grid", action="store_true", help="Generate comparison grid")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    values = [float(v) if "." in v else int(v) for v in args.values.split(",")]
    client = ComfyUIClient(args.host)
    wf_base = load_workflow(args.workflow)
    results = []

    for val in values:
        label = f"{args.param}={val}"
        print(f"Running: {label}")

        wf = parametrize_workflow(wf_base, {
            args.param: val,
            "3.seed": args.seed,
        })

        saved = client.run_workflow(
            wf,
            images={"10.image": args.input},
            output_dir=output_dir,
        )
        if saved:
            renamed = output_dir / f"sweep_{args.param.replace('.', '_')}_{val}.png"
            saved[0].rename(renamed)
            results.append((label, renamed))
            print(f"  -> {renamed.name}")

    if args.grid and results:
        grid_path = output_dir / f"grid_{args.param.replace('.', '_')}.png"
        create_comparison_grid(results, grid_path, original=Path(args.input))

    # Save sweep metadata
    meta = {
        "param": args.param,
        "values": values,
        "seed": args.seed,
        "input": args.input,
        "results": [(l, str(p)) for l, p in results],
    }
    (output_dir / "sweep.json").write_text(json.dumps(meta, indent=2))
    print(f"\nSweep complete: {len(results)} results")


if __name__ == "__main__":
    main()
