"""Dataset preparation for LoRA training.

Processes images from various simulation projects into training-ready pairs:
- Crop/resize to target resolution
- Auto-caption with Florence-2 (or manual captions)
- Add trigger word prefix
- Output: img_001.png + img_001.txt pairs

Usage:
    python prepare_dataset.py --input ../datasets/raw/dla/ --output ../datasets/dla_aesthetic/ --trigger dlaaesthetic
    python prepare_dataset.py --input ../datasets/raw/sims/ --output ../datasets/sim_aesthetic/ --trigger simaesthetic --caption auto
"""

import argparse
import json
from pathlib import Path

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    from transformers import AutoProcessor, AutoModelForCausalLM
    import torch
    HAS_FLORENCE = True
except ImportError:
    HAS_FLORENCE = False


# ── Default captions per dataset type ─────────────────────────

DEFAULT_CAPTIONS = {
    "dla": (
        "dlaaesthetic, bioluminescent dendritic crystalline structure, "
        "fractal branching pattern resembling neural networks and river deltas, "
        "scanning electron microscope aesthetic, ice blue and warm amber"
    ),
    "sim": (
        "simaesthetic, generative simulation visualization, organic computational patterns, "
        "emergent algorithmic structures, scientific visualization aesthetic"
    ),
    "physarum": (
        "simaesthetic, physarum polycephalum slime mold network, "
        "organic branching transport network, bioluminescent veins"
    ),
    "diffgrowth": (
        "simaesthetic, differential growth simulation, undulating organic edges, "
        "coral-like folding membrane structures"
    ),
}


def resize_and_crop(img: "Image.Image", size: int) -> "Image.Image":
    """Center-crop to square then resize to target size."""
    w, h = img.size
    s = min(w, h)
    left = (w - s) // 2
    top = (h - s) // 2
    img = img.crop((left, top, left + s, top + s))
    return img.resize((size, size), Image.LANCZOS)


def resize_fit(img: "Image.Image", size: int) -> "Image.Image":
    """Resize longest side to target, pad shorter side with black."""
    w, h = img.size
    ratio = size / max(w, h)
    new_w, new_h = int(w * ratio), int(h * ratio)
    img = img.resize((new_w, new_h), Image.LANCZOS)

    canvas = Image.new("RGB", (size, size), (0, 0, 0))
    canvas.paste(img, ((size - new_w) // 2, (size - new_h) // 2))
    return canvas


class Florence2Captioner:
    """Auto-caption images using Florence-2."""

    def __init__(self, model_name: str = "microsoft/Florence-2-base"):
        if not HAS_FLORENCE:
            raise ImportError("pip install transformers torch")
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
        if torch.backends.mps.is_available():
            self.model = self.model.to("mps")
        self.device = next(self.model.parameters()).device

    def caption(self, image: "Image.Image") -> str:
        """Generate detailed caption for image."""
        task = "<MORE_DETAILED_CAPTION>"
        inputs = self.processor(text=task, images=image, return_tensors="pt").to(self.device)
        generated = self.model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=128,
            num_beams=3,
        )
        result = self.processor.batch_decode(generated, skip_special_tokens=True)[0]
        # Florence returns "task_token caption", strip the task token
        if result.startswith("<MORE_DETAILED_CAPTION>"):
            result = result[len("<MORE_DETAILED_CAPTION>"):].strip()
        return result


def process_dataset(
    input_dir: Path,
    output_dir: Path,
    trigger: str,
    size: int = 1024,
    crop_mode: str = "crop",
    caption_mode: str = "default",
    dataset_type: str = "dla",
    custom_caption: str | None = None,
):
    """Process raw images into training-ready dataset."""
    if not HAS_PIL:
        raise ImportError("pip install Pillow")

    output_dir.mkdir(parents=True, exist_ok=True)
    exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}
    images = sorted(f for f in input_dir.rglob("*") if f.suffix.lower() in exts)

    if not images:
        print(f"No images found in {input_dir}")
        return

    print(f"Processing {len(images)} images -> {output_dir}")

    captioner = None
    if caption_mode == "auto":
        print("Loading Florence-2 captioner...")
        captioner = Florence2Captioner()

    resize_fn = resize_and_crop if crop_mode == "crop" else resize_fit
    manifest = []

    for i, img_path in enumerate(images):
        idx = f"{i+1:03d}"
        out_img = output_dir / f"img_{idx}.png"
        out_txt = output_dir / f"img_{idx}.txt"

        img = Image.open(img_path).convert("RGB")
        img = resize_fn(img, size)
        img.save(out_img, "PNG")

        # Generate caption
        if caption_mode == "auto" and captioner:
            caption = captioner.caption(img)
            caption = f"{trigger}, {caption}"
        elif custom_caption:
            caption = f"{trigger}, {custom_caption}"
        else:
            caption = DEFAULT_CAPTIONS.get(dataset_type, f"{trigger}, generative art visualization")

        out_txt.write_text(caption)
        manifest.append({
            "index": idx,
            "source": img_path.name,
            "caption": caption,
        })
        print(f"  [{idx}/{len(images):03d}] {img_path.name} -> img_{idx}.png")

    # Save manifest
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps({
        "dataset_type": dataset_type,
        "trigger_word": trigger,
        "image_count": len(manifest),
        "resolution": size,
        "crop_mode": crop_mode,
        "caption_mode": caption_mode,
        "entries": manifest,
    }, indent=2))
    print(f"\nDataset ready: {len(manifest)} pairs in {output_dir}")
    print(f"Manifest: {manifest_path}")


def main():
    parser = argparse.ArgumentParser(description="Prepare dataset for LoRA training")
    parser.add_argument("--input", "-i", required=True, help="Raw images directory")
    parser.add_argument("--output", "-o", required=True, help="Output dataset directory")
    parser.add_argument("--trigger", "-t", required=True, help="Trigger word (e.g. dlaaesthetic)")
    parser.add_argument("--size", type=int, default=1024, help="Target resolution (default 1024)")
    parser.add_argument("--crop", choices=["crop", "fit"], default="crop", help="Resize mode")
    parser.add_argument("--caption", choices=["default", "auto", "custom"], default="default")
    parser.add_argument("--caption-text", help="Custom caption text (used with --caption custom)")
    parser.add_argument("--type", choices=list(DEFAULT_CAPTIONS.keys()), default="dla",
                        help="Dataset type for default captions")
    args = parser.parse_args()

    process_dataset(
        input_dir=Path(args.input),
        output_dir=Path(args.output),
        trigger=args.trigger,
        size=args.size,
        crop_mode=args.crop,
        caption_mode=args.caption,
        dataset_type=args.type,
        custom_caption=args.caption_text,
    )


if __name__ == "__main__":
    main()
