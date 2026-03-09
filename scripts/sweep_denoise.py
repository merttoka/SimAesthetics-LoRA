"""Parameter sweep for ComfyUI workflows with comparison grid.

Generates N outputs from a single input image with the same seed,
varying one parameter across a range. Produces a labeled grid.

Usage:
    # Denoise sweep
    python sweep_denoise.py -i frame.png -w workflows/sdxl_img2img_lora.json \
        -p 3.denoise --range 0.4,0.9,6 --host http://192.168.0.52:8188

    # LoRA strength sweep
    python sweep_denoise.py -i frame.png -w workflows/sdxl_img2img_lora.json \
        -p 40.strength_model --values 0.5,0.7,0.8,0.9,1.0,1.2

    # CFG sweep
    python sweep_denoise.py -i frame.png -w workflows/sdxl_controlnet_lora.json \
        -p 3.cfg --range 3,12,4

    # ControlNet strength sweep
    python sweep_denoise.py -i frame.png -w workflows/sdxl_controlnet_lora.json \
        -p 20.strength --range 0.3,1.0,5
"""

import argparse
import json
from pathlib import Path

from comfyui_client import ComfyUIClient, load_workflow, parametrize_workflow

from PIL import Image, ImageDraw, ImageFont


def get_font(size: int = 18):
    for name in ["/System/Library/Fonts/Helvetica.ttc", "arial.ttf"]:
        try:
            return ImageFont.truetype(name, size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


def create_sweep_grid(
    original: Path,
    results: list[tuple[str, Path]],
    output_path: Path,
    cell_size: int = 512,
    param_name: str = "",
):
    """Create labeled grid: original on the left, sweep results flowing right."""
    gap = 8
    margin = 16
    header_h = 32
    n = len(results) + 1  # +1 for original

    total_w = margin + (cell_size + gap) * n - gap + margin
    total_h = margin + header_h + cell_size + margin

    grid = Image.new("RGB", (total_w, total_h), (0, 0, 0))
    draw = ImageDraw.Draw(grid)
    font = get_font(16)

    # Original
    orig_img = Image.open(original).convert("RGB").resize((cell_size, cell_size), Image.LANCZOS)
    x = margin
    grid.paste(orig_img, (x, margin + header_h))
    draw.text((x + cell_size // 2, margin + 4), "original", fill=(180, 180, 180), font=font, anchor="mt")

    # Sweep results
    for i, (label, path) in enumerate(results):
        x = margin + (i + 1) * (cell_size + gap)
        img = Image.open(path).convert("RGB").resize((cell_size, cell_size), Image.LANCZOS)
        grid.paste(img, (x, margin + header_h))
        draw.text((x + cell_size // 2, margin + 4), label, fill=(180, 180, 180), font=font, anchor="mt")

    grid.save(output_path, "PNG")
    print(f"Grid saved: {output_path} ({grid.size[0]}x{grid.size[1]})")


def parse_range(range_str: str) -> list[float]:
    """Parse 'start,end,steps' into evenly spaced values."""
    parts = range_str.split(",")
    start, end, steps = float(parts[0]), float(parts[1]), int(parts[2])
    if steps == 1:
        return [start]
    step = (end - start) / (steps - 1)
    return [round(start + i * step, 3) for i in range(steps)]


def main():
    parser = argparse.ArgumentParser(description="Parameter sweep with comparison grid")
    parser.add_argument("--input", "-i", required=True, help="Input image")
    parser.add_argument("--workflow", "-w", required=True, help="Workflow JSON")
    parser.add_argument("--output", "-o", default="outputs/sweep", help="Output dir")
    parser.add_argument("--host", default="http://127.0.0.1:8188")
    parser.add_argument("--param", "-p", required=True, help="Parameter to sweep (node_id.field)")
    parser.add_argument("--values", "-v", help="Comma-separated values (e.g. 0.5,0.7,0.9)")
    parser.add_argument("--range", "-R", help="Range as start,end,steps (e.g. 0.4,0.9,6)")
    parser.add_argument("--seed", type=int, default=42, help="Fixed seed for comparison")
    parser.add_argument("--size", type=int, default=512, help="Grid cell size")
    parser.add_argument("--no-grid", action="store_true", help="Skip grid generation")
    args = parser.parse_args()

    if args.values:
        values = [float(v) if "." in v else int(v) for v in args.values.split(",")]
    elif args.range:
        values = parse_range(args.range)
    else:
        parser.error("Provide --values or --range")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    param_short = args.param.replace(".", "_")
    client = ComfyUIClient(args.host)
    wf_base = load_workflow(args.workflow)
    results = []

    print(f"Sweeping {args.param}: {values}")
    print(f"Seed: {args.seed}, Input: {args.input}")

    for val in values:
        label = f"{val}"
        print(f"  [{len(results)+1}/{len(values)}] {args.param}={val}...")

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
            renamed = output_dir / f"sweep_{param_short}_{val}.png"
            saved[0].rename(renamed)
            results.append((label, renamed))

    if not args.no_grid and results:
        grid_path = output_dir / f"grid_{param_short}.png"
        create_sweep_grid(
            Path(args.input), results, grid_path,
            cell_size=args.size, param_name=args.param,
        )

    meta = {
        "param": args.param,
        "values": values,
        "seed": args.seed,
        "input": args.input,
        "workflow": args.workflow,
        "results": [(l, str(p)) for l, p in results],
    }
    (output_dir / "sweep.json").write_text(json.dumps(meta, indent=2))
    print(f"\nDone: {len(results)} results")


if __name__ == "__main__":
    main()
