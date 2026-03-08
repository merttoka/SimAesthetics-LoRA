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
import random
import subprocess
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
    "biomass_growth": (
        "biomaesthetic, pristine cellular growth, lush bioluminescent tissue, "
        "stage 1 biomass, vibrant living organic matter, dense healthy cellular structure, "
        "scanning electron microscope aesthetic"
    ),
    "biomass_mature": (
        "biomaesthetic, mature biological structure, established organic network, "
        "stage 2 biomass, complex interwoven tissue, peak biological density"
    ),
    "biomass_aging": (
        "biomaesthetic, aging biological matter, early decomposition, "
        "stage 3 biomass, fading cellular integrity, transitional organic decay"
    ),
    "biomass_decay": (
        "biomaesthetic, decaying organic matter, fungal colonization, "
        "stage 4 biomass, necrotic tissue breakdown, dissolving cellular membranes, "
        "rot and spore formation"
    ),
    "biomass_necrotic": (
        "biomaesthetic, necrotic dissolving structures, advanced decomposition, "
        "stage 5 biomass, primordial organic dissolution, entropic biological collapse"
    ),
    "sem": (
        "biomaesthetic, scanning electron microscope image, extreme magnification, "
        "false-color enhancement, microscopic biological surface detail"
    ),
    "fungal": (
        "biomaesthetic, fungal mycelium network, branching hyphae, "
        "spore-bearing structures, bioluminescent fungal growth"
    ),
    "rust": (
        "biomaesthetic, oxidized metal surface, iron rust texture, "
        "corrosion patterns, chemical decay patina"
    ),
}

# ── Biomass lifecycle stage definitions ──────────────────────

BIOMASS_STAGES = {
    1: "pristine, vibrant, dense cellular growth",
    2: "mature, established, peak density",
    3: "aging, transitional, early decay",
    4: "decaying, necrotic, fungal colonization",
    5: "dissolved, entropic, primordial collapse",
}


def interpolate_stage_caption(
    index: int, total: int, start_stage: int, end_stage: int, trigger: str
) -> str:
    """Return a stage-interpolated caption based on position in sequence.

    Maps index (0-based) within total images to the nearest lifecycle stage
    between start_stage and end_stage, producing a caption with stage number.
    """
    if total <= 1:
        t = 0.0
    else:
        t = index / (total - 1)

    stage_float = start_stage + t * (end_stage - start_stage)
    stage = round(stage_float)
    stage = max(min(stage, 5), 1)

    desc = BIOMASS_STAGES[stage]
    return (
        f"{trigger}, biomaesthetic, stage {stage} biomass, {desc}"
    )


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


def _mean_brightness(img: "Image.Image") -> float:
    """Average pixel brightness (0-255) of an RGB image."""
    gray = img.convert("L")
    pixels = gray.getdata()
    return sum(pixels) / len(pixels)


def sample_regions(
    img: "Image.Image", size: int, count: int = 3,
    h_focus: float = 0.6, v_focus: float = 1.0,
    min_brightness: float = 8.0, max_attempts: int = 50,
) -> list["Image.Image"]:
    """Sample random square crops from the focus region of a wide image.

    For ultrawide sim frames (e.g. 14336x1920), focuses on the middle portion
    of horizontal space where the sim content lives, avoiding empty edges.
    Rejects crops that are mostly black (below min_brightness threshold).

    Args:
        img: Source image (any aspect ratio)
        size: Output square size (e.g. 1024)
        count: Number of crops to extract per image
        h_focus: Fraction of horizontal center to sample from (0.3 = middle 30%)
        v_focus: Fraction of vertical center to sample from (1.0 = full height)
        min_brightness: Reject crops with mean brightness below this (0-255)
        max_attempts: Max random tries before giving up on an image
    """
    w, h = img.size
    crop_s = min(h, w)  # crop square can't exceed shorter dimension

    # Define the focus region
    h_margin = int(w * (1 - h_focus) / 2)
    v_margin = int(h * (1 - v_focus) / 2)
    # Clamp valid crop origins so the crop_s square fits within image bounds
    x_min = h_margin
    x_max = max(x_min, w - h_margin - crop_s)
    y_min = v_margin
    y_max = max(y_min, h - v_margin - crop_s)

    crops = []
    attempts = 0
    while len(crops) < count and attempts < max_attempts:
        x = random.randint(x_min, x_max)
        y = random.randint(y_min, y_max)
        cropped = img.crop((x, y, x + crop_s, y + crop_s))
        attempts += 1

        if _mean_brightness(cropped) < min_brightness:
            continue

        crops.append(cropped.resize((size, size), Image.LANCZOS))
    return crops


def extract_video_frames(
    video_path: Path, output_dir: Path, interval: float = 1.0,
    skip_start: float = 5.0,
) -> list[Path]:
    """Extract frames from MP4 at regular intervals using ffmpeg.

    Args:
        video_path: Path to video file
        output_dir: Where to save extracted PNGs
        interval: Seconds between extracted frames
        skip_start: Skip this many seconds from the start (sim may be empty)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(skip_start),
        "-i", str(video_path),
        "-vf", f"fps=1/{interval}",
        str(output_dir / "vframe_%04d.png"),
    ]
    print(f"Extracting frames from {video_path.name} (interval={interval}s, skip={skip_start}s)...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ffmpeg error: {result.stderr[-500:]}")
        return []
    frames = sorted(output_dir.glob("vframe_*.png"))
    print(f"  Extracted {len(frames)} frames")
    return frames


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
    stage_range: tuple[int, int] | None = None,
    samples_per_image: int = 3,
    h_focus: float = 0.6,
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
    if crop_mode == "sample":
        print(f"  Sampling {samples_per_image} crops/image, h_focus={h_focus}")

    captioner = None
    if caption_mode == "auto":
        print("Loading Florence-2 captioner...")
        captioner = Florence2Captioner()

    manifest = []
    img_counter = 0

    for i, img_path in enumerate(images):
        img = Image.open(img_path).convert("RGB")

        # Generate crops
        if crop_mode == "sample":
            crops = sample_regions(img, size, samples_per_image, h_focus)
        elif crop_mode == "fit":
            crops = [resize_fit(img, size)]
        else:
            crops = [resize_and_crop(img, size)]

        for ci, crop in enumerate(crops):
            img_counter += 1
            idx = f"{img_counter:03d}"
            out_img = output_dir / f"img_{idx}.png"
            out_txt = output_dir / f"img_{idx}.txt"

            crop.save(out_img, "PNG")

            # Generate caption
            if stage_range:
                caption = interpolate_stage_caption(
                    i, len(images), stage_range[0], stage_range[1], trigger
                )
            elif caption_mode == "auto" and captioner:
                caption = captioner.caption(crop)
                caption = f"{trigger}, {caption}"
            elif custom_caption:
                caption = f"{trigger}, {custom_caption}"
            else:
                caption = DEFAULT_CAPTIONS.get(dataset_type, f"{trigger}, generative art visualization")

            out_txt.write_text(caption)
            manifest.append({
                "index": idx,
                "source": img_path.name,
                "crop": ci if crop_mode == "sample" else None,
                "caption": caption,
            })

        suffix = f" ({len(crops)} crops)" if crop_mode == "sample" else ""
        print(f"  [{i+1}/{len(images)}] {img_path.name}{suffix}")

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
    parser.add_argument("--crop", choices=["crop", "fit", "sample"], default="crop",
                        help="Resize mode: crop (center), fit (pad), sample (random regions from wide images)")
    parser.add_argument("--samples", type=int, default=3,
                        help="Crops per image in sample mode (default: 3)")
    parser.add_argument("--h-focus", type=float, default=0.6,
                        help="Horizontal focus region for sample mode (0.6 = middle 60%%)")
    parser.add_argument("--caption", choices=["default", "auto", "custom"], default="default")
    parser.add_argument("--caption-text", help="Custom caption text (used with --caption custom)")
    parser.add_argument("--type", choices=list(DEFAULT_CAPTIONS.keys()), default="dla",
                        help="Dataset type for default captions")
    parser.add_argument("--stage-range", help="Interpolate biomass stages across sequence (e.g. 1,5)")
    parser.add_argument("--video", help="Extract frames from MP4 first (provide video path)")
    parser.add_argument("--video-interval", type=float, default=1.0,
                        help="Seconds between video frame extracts (default: 1.0)")
    parser.add_argument("--video-skip", type=float, default=5.0,
                        help="Skip N seconds from video start (default: 5.0)")
    args = parser.parse_args()

    # If --video provided, extract frames into input dir first
    input_dir = Path(args.input)
    if args.video:
        input_dir.mkdir(parents=True, exist_ok=True)
        extract_video_frames(
            Path(args.video), input_dir,
            interval=args.video_interval, skip_start=args.video_skip,
        )

    stage_range = None
    if args.stage_range:
        parts = args.stage_range.split(",")
        stage_range = (int(parts[0]), int(parts[1]))

    process_dataset(
        input_dir=input_dir,
        output_dir=Path(args.output),
        trigger=args.trigger,
        size=args.size,
        crop_mode=args.crop,
        caption_mode=args.caption,
        dataset_type=args.type,
        custom_caption=args.caption_text,
        stage_range=stage_range,
        samples_per_image=args.samples,
        h_focus=args.h_focus,
    )


if __name__ == "__main__":
    main()
