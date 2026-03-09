"""Parameter sweep for ComfyUI workflows with comparison grid.

Generates N outputs from a single input image with the same seed,
varying one or two parameters across a range. Produces a labeled grid.

Usage:
    # Single param sweep (1D grid)
    python sweep_denoise.py -i frame.png -w workflows/sdxl_img2img_lora.json \
        -p 3.denoise --range 0.4,0.9,6 --host http://<comfyui-host>:8188

    # Multi-param sweep (2D grid: rows × columns)
    python sweep_denoise.py -i frame.png -w workflows/sdxl_controlnet_lora.json \
        -p 40.strength_model --range 0.15,0.5,4 \
        -p2 20.strength --range2 0.3,0.8,4

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


def create_2d_sweep_grid(
    original: Path,
    results: dict[tuple[float, float], Path],
    p1_values: list[float],
    p2_values: list[float],
    p1_name: str,
    p2_name: str,
    output_path: Path,
    cell_size: int = 512,
):
    """Create 2D grid: rows = param1, columns = param2, with axis labels."""
    gap = 4
    margin = 16
    label_w = 80   # left column for row labels
    header_h = 48  # top row for column labels

    cols = len(p2_values)
    rows = len(p1_values)

    total_w = margin + label_w + (cell_size + gap) * cols - gap + margin
    total_h = margin + header_h + (cell_size + gap) * rows - gap + margin

    grid = Image.new("RGB", (total_w, total_h), (0, 0, 0))
    draw = ImageDraw.Draw(grid)
    font = get_font(14)
    font_small = get_font(12)

    # Axis labels
    p1_short = p1_name.split(".")[-1]
    p2_short = p2_name.split(".")[-1]

    # Column headers (param2 values)
    for j, v2 in enumerate(p2_values):
        x = margin + label_w + j * (cell_size + gap) + cell_size // 2
        draw.text((x, margin + 4), f"{p2_short}={v2}", fill=(180, 180, 180), font=font, anchor="mt")

    # Row labels + cells
    for i, v1 in enumerate(p1_values):
        y = margin + header_h + i * (cell_size + gap)
        # Row label
        draw.text((margin + label_w - 8, y + cell_size // 2), f"{p1_short}={v1}", fill=(180, 180, 180), font=font_small, anchor="rm")
        for j, v2 in enumerate(p2_values):
            x = margin + label_w + j * (cell_size + gap)
            key = (v1, v2)
            if key in results and results[key].exists():
                img = Image.open(results[key]).convert("RGB").resize((cell_size, cell_size), Image.LANCZOS)
                grid.paste(img, (x, y))

    grid.save(output_path, "PNG")
    print(f"2D Grid saved: {output_path} ({grid.size[0]}x{grid.size[1]}, {rows}x{cols})")


def parse_range(range_str: str) -> list[float]:
    """Parse 'start,end,steps' into evenly spaced values."""
    parts = range_str.split(",")
    start, end, steps = float(parts[0]), float(parts[1]), int(parts[2])
    if steps == 1:
        return [start]
    step = (end - start) / (steps - 1)
    return [round(start + i * step, 3) for i in range(steps)]


def parse_values(values_str: str) -> list[float]:
    return [float(v) if "." in v else int(v) for v in values_str.split(",")]


def main():
    parser = argparse.ArgumentParser(description="Parameter sweep with comparison grid")
    parser.add_argument("--input", "-i", required=True, help="Input image")
    parser.add_argument("--workflow", "-w", required=True, help="Workflow JSON")
    parser.add_argument("--output", "-o", default="outputs/sweep", help="Output dir")
    parser.add_argument("--host", default="http://127.0.0.1:8188")
    parser.add_argument("--param", "-p", required=True, help="Parameter to sweep (node_id.field)")
    parser.add_argument("--values", "-v", help="Comma-separated values (e.g. 0.5,0.7,0.9)")
    parser.add_argument("--range", "-R", help="Range as start,end,steps (e.g. 0.4,0.9,6)")
    parser.add_argument("--param2", "-p2", help="Second parameter for 2D sweep")
    parser.add_argument("--values2", "-v2", help="Comma-separated values for param2")
    parser.add_argument("--range2", "-R2", help="Range for param2 as start,end,steps")
    parser.add_argument("--seed", type=int, default=42, help="Fixed seed for comparison")
    parser.add_argument("--size", type=int, default=512, help="Grid cell size")
    parser.add_argument("--no-grid", action="store_true", help="Skip grid generation")
    args = parser.parse_args()

    if args.values:
        values1 = parse_values(args.values)
    elif args.range:
        values1 = parse_range(args.range)
    else:
        parser.error("Provide --values or --range")

    is_2d = args.param2 is not None
    values2 = None
    if is_2d:
        if args.values2:
            values2 = parse_values(args.values2)
        elif args.range2:
            values2 = parse_range(args.range2)
        else:
            parser.error("Provide --values2 or --range2 for second parameter")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    p1_short = args.param.replace(".", "_")
    client = ComfyUIClient(args.host)
    wf_base = load_workflow(args.workflow)

    if is_2d:
        # 2D sweep
        p2_short = args.param2.replace(".", "_")
        total = len(values1) * len(values2)
        results_2d = {}
        count = 0

        print(f"2D sweep: {args.param} × {args.param2}")
        print(f"  {args.param}: {values1}")
        print(f"  {args.param2}: {values2}")
        print(f"  Total: {total} runs, Seed: {args.seed}")

        for v1 in values1:
            for v2 in values2:
                count += 1
                print(f"  [{count}/{total}] {args.param}={v1}, {args.param2}={v2}...")

                wf = parametrize_workflow(wf_base, {
                    args.param: v1,
                    args.param2: v2,
                    "3.seed": args.seed,
                })

                saved = client.run_workflow(
                    wf,
                    images={"10.image": args.input},
                    output_dir=output_dir,
                )
                if saved:
                    renamed = output_dir / f"sweep_{p1_short}_{v1}_{p2_short}_{v2}.png"
                    saved[0].rename(renamed)
                    results_2d[(v1, v2)] = renamed

        if not args.no_grid and results_2d:
            grid_path = output_dir / f"grid_{p1_short}_x_{p2_short}.png"
            create_2d_sweep_grid(
                Path(args.input), results_2d,
                values1, values2,
                args.param, args.param2,
                grid_path, cell_size=args.size,
            )

        meta = {
            "param1": args.param,
            "values1": values1,
            "param2": args.param2,
            "values2": values2,
            "seed": args.seed,
            "input": args.input,
            "workflow": args.workflow,
            "results": {f"{v1},{v2}": str(p) for (v1, v2), p in results_2d.items()},
        }
        (output_dir / "sweep.json").write_text(json.dumps(meta, indent=2))
        print(f"\nDone: {len(results_2d)} results ({len(values1)}x{len(values2)})")

    else:
        # 1D sweep
        results = []
        print(f"Sweeping {args.param}: {values1}")
        print(f"Seed: {args.seed}, Input: {args.input}")

        for val in values1:
            label = f"{val}"
            print(f"  [{len(results)+1}/{len(values1)}] {args.param}={val}...")

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
                renamed = output_dir / f"sweep_{p1_short}_{val}.png"
                saved[0].rename(renamed)
                results.append((label, renamed))

        if not args.no_grid and results:
            grid_path = output_dir / f"grid_{p1_short}.png"
            create_sweep_grid(
                Path(args.input), results, grid_path,
                cell_size=args.size, param_name=args.param,
            )

        meta = {
            "param": args.param,
            "values": values1,
            "seed": args.seed,
            "input": args.input,
            "workflow": args.workflow,
            "results": [(l, str(p)) for l, p in results],
        }
        (output_dir / "sweep.json").write_text(json.dumps(meta, indent=2))
        print(f"\nDone: {len(results)} results")


if __name__ == "__main__":
    main()
