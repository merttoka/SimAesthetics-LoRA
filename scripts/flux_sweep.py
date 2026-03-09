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


def main():
    parser = argparse.ArgumentParser(description="FLUX img2img parameter sweeps")
    parser.add_argument("--input", "-i", required=True, help="Input image")
    parser.add_argument("--lora", default=None, help="LoRA path (auto-detect)")
    parser.add_argument("--output", "-o", default="/workspace/flux_sweeps", help="Output dir")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sweep", nargs="+", default=["denoise", "lora", "steps"],
                        choices=["denoise", "lora", "steps"],
                        help="Which sweeps to run (default: all 3)")
    parser.add_argument("--values", type=str, default=None,
                        help="Custom values for first sweep (comma-separated)")
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

    for sweep_name in args.sweep:
        if args.values and sweep_name == args.sweep[0]:
            values = [float(v) for v in args.values.split(",")]
        else:
            values = SWEEPS[sweep_name]

        images, labels = run_sweep(pipe, input_image, sweep_name, values, output_dir, args.seed)

        # Build title showing fixed params
        fixed = {k: v for k, v in DEFAULTS.items()}
        title = f"FLUX {sweep_name} sweep"
        grid = make_grid(images, labels, input_image, title)
        grid_path = output_dir / f"grid_sweep_{sweep_name}.png"
        grid.save(grid_path)
        print(f"  Grid: {grid_path}")

    print(f"\nAll sweeps done → {output_dir}")


if __name__ == "__main__":
    main()
