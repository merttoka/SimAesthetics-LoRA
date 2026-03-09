"""FLUX parameter sweeps for img2img structural fidelity.

Runs sweeps on a single input frame to find the best params for
preserving simulation structure while adding organic texture.

Sweeps:
  1. denoise (strength): how much structure survives the round-trip
  2. lora: aesthetic intensity vs base model freedom
  3. steps: inference quality/fidelity tradeoff

Generates labeled grid per sweep. Designed for RunPod A100.

Usage:
    # All 3 sweeps with defaults
    python flux_sweep.py -i /workspace/datasets/sim_aesthetic_2/img_010.png

    # Single sweep with custom values
    python flux_sweep.py -i frame.png --sweep denoise --values 0.3,0.4,0.5,0.6

    # All sweeps, custom seed
    python flux_sweep.py -i frame.png --seed 123
"""

import argparse
from pathlib import Path

import torch
from diffusers import FluxImg2ImgPipeline
from PIL import Image, ImageDraw, ImageFont


PROMPT = "simaesthetic, bioluminescent organic network, physarum slime mold veins, cyan and amber glow, dark background"

DEFAULTS = {
    "denoise": 0.7,
    "lora_strength": 0.35,
    "steps": 28,
    "cfg": 3.5,
}

SWEEPS = {
    "denoise": [0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
    "lora": [0.0, 0.15, 0.25, 0.35, 0.5, 0.7, 0.9],
    "steps": [10, 15, 20, 28, 40, 50],
}


def find_lora(base_dir="/workspace/output/sim_aesthetic_flux/sim_aesthetic_flux"):
    base = Path(base_dir)
    final = base / "sim_aesthetic_flux.safetensors"
    if final.exists():
        return str(final)
    checkpoints = sorted(base.glob("*.safetensors"))
    if checkpoints:
        return str(checkpoints[-1])
    raise FileNotFoundError(f"No LoRA found in {base}")


def make_grid(images, labels, input_img, title, cell_size=512):
    """Labeled grid: input on left, sweep results right."""
    n = len(images)
    cols = n + 1
    label_h = 40
    title_h = 36
    w = cols * cell_size
    h = cell_size + label_h + title_h

    grid = Image.new("RGB", (w, h), (0, 0, 0))
    draw = ImageDraw.Draw(grid)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
    except OSError:
        font = ImageFont.load_default()

    # Title
    draw.text((w // 2, 10), title, fill=(255, 220, 100), font=font, anchor="mt")

    y_top = title_h

    # Input
    resized_input = input_img.resize((cell_size, cell_size), Image.LANCZOS)
    grid.paste(resized_input, (0, y_top + label_h))
    draw.text((cell_size // 2, y_top + 10), "input", fill=(150, 150, 150), font=font, anchor="mt")

    # Sweep results
    for i, (img, label) in enumerate(zip(images, labels)):
        x = (i + 1) * cell_size
        resized = img.resize((cell_size, cell_size), Image.LANCZOS)
        grid.paste(resized, (x, y_top + label_h))
        draw.text((x + cell_size // 2, y_top + 10), str(label),
                   fill=(255, 255, 255), font=font, anchor="mt")

    return grid


def run_sweep(pipe, input_image, sweep_name, values, output_dir, seed):
    """Run one sweep, return images and labels."""
    print(f"\n{'='*60}")
    print(f"  SWEEP: {sweep_name} = {values}")
    print(f"  Fixed: denoise={DEFAULTS['denoise']}, lora={DEFAULTS['lora_strength']}, "
          f"steps={DEFAULTS['steps']}, cfg={DEFAULTS['cfg']}")
    print(f"{'='*60}")

    images = []
    labels = []

    for i, val in enumerate(values):
        denoise = DEFAULTS["denoise"]
        lora_strength = DEFAULTS["lora_strength"]
        steps = DEFAULTS["steps"]
        cfg = DEFAULTS["cfg"]

        if sweep_name == "denoise":
            denoise = val
            label = f"{val:.2f}"
        elif sweep_name == "lora":
            lora_strength = val
            label = f"{val:.2f}"
        elif sweep_name == "steps":
            steps = int(val)
            label = str(steps)

        gen = torch.Generator("cuda").manual_seed(seed)

        print(f"  [{i+1}/{len(values)}] {sweep_name}={label} "
              f"(d={denoise}, lora={lora_strength}, s={steps})")

        result = pipe(
            prompt=PROMPT,
            image=input_image,
            strength=denoise,
            guidance_scale=cfg,
            num_inference_steps=steps,
            generator=gen,
            joint_attention_kwargs={"scale": lora_strength},
        ).images[0]

        out_path = output_dir / f"sweep_{sweep_name}_{label}.png"
        result.save(out_path)
        images.append(result)
        labels.append(label)

    return images, labels


def run_2d_sweep(pipe, input_image, name1, values1, name2, values2, output_dir, seed):
    """Run 2D sweep: rows=param1, cols=param2. Return grid of images."""
    print(f"\n{'='*60}")
    print(f"  2D SWEEP: {name1} × {name2}")
    print(f"  {name1} = {values1}")
    print(f"  {name2} = {values2}")
    print(f"  Total: {len(values1) * len(values2)} images")
    print(f"{'='*60}")

    grid_images = []  # list of rows, each row is list of images

    for i, v1 in enumerate(values1):
        row = []
        for j, v2 in enumerate(values2):
            denoise = DEFAULTS["denoise"]
            lora_strength = DEFAULTS["lora_strength"]
            steps = DEFAULTS["steps"]
            cfg = DEFAULTS["cfg"]

            for name, val in [(name1, v1), (name2, v2)]:
                if name == "denoise":
                    denoise = val
                elif name == "lora":
                    lora_strength = val
                elif name == "steps":
                    steps = int(val)

            gen = torch.Generator("cuda").manual_seed(seed)
            idx = i * len(values2) + j + 1
            total = len(values1) * len(values2)
            print(f"  [{idx}/{total}] {name1}={v1}, {name2}={v2} "
                  f"(d={denoise}, lora={lora_strength}, s={steps})")

            result = pipe(
                prompt=PROMPT, image=input_image, strength=denoise,
                guidance_scale=cfg, num_inference_steps=steps,
                generator=gen, joint_attention_kwargs={"scale": lora_strength},
            ).images[0]

            fmt1 = f"{v1:.2f}" if isinstance(v1, float) else str(int(v1))
            fmt2 = f"{v2:.2f}" if isinstance(v2, float) else str(int(v2))
            out_path = output_dir / f"sweep_{name1}{fmt1}_{name2}{fmt2}.png"
            result.save(out_path)
            row.append(result)

        grid_images.append(row)

    return grid_images


def make_2d_grid(grid_images, row_labels, col_labels, row_name, col_name,
                 input_img, cell_size=384):
    """Create labeled 2D matrix grid with row/col axis labels."""
    rows = len(grid_images)
    cols = len(grid_images[0])
    label_w = 80   # left margin for row labels
    label_h = 50   # top margin for col labels
    title_h = 36

    w = label_w + cols * cell_size
    h = title_h + label_h + rows * cell_size

    grid = Image.new("RGB", (w, h), (0, 0, 0))
    draw = ImageDraw.Draw(grid)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
        font_sm = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except OSError:
        font = ImageFont.load_default()
        font_sm = font

    # Title
    title = f"FLUX 2D: {row_name} (rows) × {col_name} (cols)"
    draw.text((w // 2, 10), title, fill=(255, 220, 100), font=font, anchor="mt")

    # Column labels
    for j, cl in enumerate(col_labels):
        x = label_w + j * cell_size + cell_size // 2
        draw.text((x, title_h + 15), str(cl), fill=(255, 255, 255), font=font, anchor="mt")

    # Col axis name
    draw.text((label_w + cols * cell_size // 2, title_h + 2), col_name,
              fill=(150, 150, 150), font=font_sm, anchor="mt")

    # Row labels + images
    for i, (row, rl) in enumerate(zip(grid_images, row_labels)):
        y = title_h + label_h + i * cell_size
        # Row label
        draw.text((label_w // 2, y + cell_size // 2), str(rl),
                  fill=(255, 255, 255), font=font, anchor="mm")
        for j, img in enumerate(row):
            x = label_w + j * cell_size
            resized = img.resize((cell_size, cell_size), Image.LANCZOS)
            grid.paste(resized, (x, y))

    # Row axis name (vertical)
    draw.text((8, title_h + label_h + rows * cell_size // 2), row_name,
              fill=(150, 150, 150), font=font_sm, anchor="lm")

    return grid


def parse_range(s):
    """Parse 'start,end,steps' into list of floats."""
    parts = s.split(",")
    start, end, n = float(parts[0]), float(parts[1]), int(parts[2])
    if n == 1:
        return [start]
    step = (end - start) / (n - 1)
    return [round(start + i * step, 4) for i in range(n)]


def main():
    parser = argparse.ArgumentParser(description="FLUX img2img parameter sweeps (1D or 2D)")
    parser.add_argument("--input", "-i", required=True, help="Input image")
    parser.add_argument("--lora", default=None, help="LoRA path (auto-detect)")
    parser.add_argument("--output", "-o", default="/workspace/flux_sweeps", help="Output dir")
    parser.add_argument("--seed", type=int, default=42)
    # 1D sweep args
    parser.add_argument("--sweep", nargs="+", default=None,
                        choices=["denoise", "lora", "steps"],
                        help="1D sweep(s) to run")
    parser.add_argument("--values", type=str, default=None,
                        help="Custom values for first 1D sweep (comma-separated)")
    # 2D sweep args
    parser.add_argument("-p", "--param", choices=["denoise", "lora", "steps"],
                        help="2D sweep: first param (rows)")
    parser.add_argument("-p2", "--param2", choices=["denoise", "lora", "steps"],
                        help="2D sweep: second param (cols)")
    parser.add_argument("--range", dest="range1", type=str,
                        help="Range for param: start,end,steps")
    parser.add_argument("--range2", type=str,
                        help="Range for param2: start,end,steps")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    input_image = Image.open(args.input).convert("RGB").resize((1024, 1024))

    print("Loading FLUX img2img pipeline...")
    pipe = FluxImg2ImgPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16
    )
    lora_path = args.lora or find_lora()
    print(f"Loading LoRA: {lora_path}")
    pipe.load_lora_weights(lora_path)
    pipe.to("cuda")
    print("Pipeline ready\n")

    # 2D sweep mode
    if args.param and args.param2:
        values1 = parse_range(args.range1)
        values2 = parse_range(args.range2)

        grid_images = run_2d_sweep(
            pipe, input_image, args.param, values1, args.param2, values2,
            output_dir, args.seed
        )

        row_labels = [f"{v:.2f}" if isinstance(v, float) else str(int(v)) for v in values1]
        col_labels = [f"{v:.2f}" if isinstance(v, float) else str(int(v)) for v in values2]

        grid = make_2d_grid(grid_images, row_labels, col_labels,
                            args.param, args.param2, input_image)
        grid_path = output_dir / f"grid_2d_{args.param}_x_{args.param2}.png"
        grid.save(grid_path)
        print(f"  Grid: {grid_path}")

    # 1D sweep mode
    else:
        sweep_names = args.sweep or ["denoise", "lora", "steps"]
        for sweep_name in sweep_names:
            if args.values and sweep_name == sweep_names[0]:
                values = [float(v) for v in args.values.split(",")]
            else:
                values = SWEEPS[sweep_name]

            images, labels = run_sweep(pipe, input_image, sweep_name, values, output_dir, args.seed)

            title = f"FLUX {sweep_name} sweep"
            grid = make_grid(images, labels, input_image, title)
            grid_path = output_dir / f"grid_sweep_{sweep_name}.png"
            grid.save(grid_path)
            print(f"  Grid: {grid_path}")

    print(f"\nAll sweeps done → {output_dir}")


if __name__ == "__main__":
    main()
