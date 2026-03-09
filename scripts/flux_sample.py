"""Generate sample images from a trained FLUX LoRA using diffusers.

Runs directly on GPU — no ComfyUI needed. Designed for RunPod A100.

Usage:
    # Basic (uses defaults)
    python flux_sample.py

    # Custom LoRA + output
    python flux_sample.py --lora /workspace/output/.../my_lora.safetensors \
        --output /workspace/samples/ --lora-strength 0.8

    # img2img mode
    python flux_sample.py --input frame.png --denoise 0.7

    # Custom prompts
    python flux_sample.py --prompts "simaesthetic, coral reef" "simaesthetic, mycelium network"
"""

import argparse
from pathlib import Path

import torch
from diffusers import FluxPipeline, FluxImg2ImgPipeline
from PIL import Image


def find_lora(base_dir: str = "/workspace/output/sim_aesthetic_flux/sim_aesthetic_flux") -> str:
    """Find the final LoRA checkpoint."""
    base = Path(base_dir)
    # Final checkpoint (no step number)
    final = base / "sim_aesthetic_flux.safetensors"
    if final.exists():
        return str(final)
    # Fall back to highest step checkpoint
    checkpoints = sorted(base.glob("*.safetensors"))
    if checkpoints:
        return str(checkpoints[-1])
    raise FileNotFoundError(f"No LoRA checkpoints found in {base}")


def main():
    parser = argparse.ArgumentParser(description="FLUX LoRA sample generation")
    parser.add_argument("--lora", default=None, help="Path to LoRA safetensors (auto-detects if omitted)")
    parser.add_argument("--lora-strength", type=float, default=0.9, help="LoRA strength")
    parser.add_argument("--output", "-o", default="/workspace/flux_samples", help="Output directory")
    parser.add_argument("--input", "-i", default=None, help="Input image for img2img mode")
    parser.add_argument("--denoise", type=float, default=0.7, help="Denoise strength for img2img")
    parser.add_argument("--prompts", nargs="+", default=None, help="Custom prompts")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--steps", type=int, default=28, help="Inference steps")
    parser.add_argument("--cfg", type=float, default=3.5, help="Guidance scale")
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--no-lora", action="store_true", help="Generate without LoRA (baseline)")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    default_prompts = [
        "simaesthetic, bioluminescent organic network, physarum slime mold veins, cyan and amber glow, dark background",
        "simaesthetic, dense cellular growth structure, branching transport network, scanning electron microscope aesthetic",
        "simaesthetic, coral reef ecosystem, living tissue, volumetric light",
        "a photograph of a golden retriever sitting in a park",
    ]
    prompts = args.prompts or default_prompts

    is_img2img = args.input is not None

    # Load pipeline
    print("Loading FLUX pipeline...")
    if is_img2img:
        pipe = FluxImg2ImgPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16
        )
    else:
        pipe = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16
        )

    # Load LoRA
    if not args.no_lora:
        lora_path = args.lora or find_lora()
        print(f"Loading LoRA: {lora_path} (strength={args.lora_strength})")
        pipe.load_lora_weights(lora_path)

    pipe.to("cuda")
    print("Pipeline ready")

    # Load input image for img2img
    input_image = None
    if is_img2img:
        input_image = Image.open(args.input).convert("RGB").resize((args.width, args.height))
        print(f"img2img mode: {args.input}, denoise={args.denoise}")

    # Generate
    generator = torch.Generator("cuda").manual_seed(args.seed)

    for i, prompt in enumerate(prompts):
        print(f"  [{i+1}/{len(prompts)}] {prompt[:60]}...")

        if is_img2img:
            result = pipe(
                prompt=prompt,
                image=input_image,
                strength=args.denoise,
                guidance_scale=args.cfg,
                num_inference_steps=args.steps,
                generator=generator,
            ).images[0]
        else:
            kwargs = dict(
                prompt=prompt,
                guidance_scale=args.cfg,
                num_inference_steps=args.steps,
                width=args.width,
                height=args.height,
                generator=generator,
            )
            if not args.no_lora:
                kwargs["joint_attention_kwargs"] = {"scale": args.lora_strength}
            result = pipe(**kwargs).images[0]

        out_path = output_dir / f"flux_sample_{i:03d}.png"
        result.save(out_path)
        print(f"    Saved: {out_path}")

    print(f"\nDone: {len(prompts)} samples in {output_dir}")


if __name__ == "__main__":
    main()
