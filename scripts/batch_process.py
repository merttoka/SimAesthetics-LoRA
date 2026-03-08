"""Batch process DLA frames through ComfyUI.

Mirrors the parallel/chained transform modes from EmergentWorlds.tsx.
Runs from Mac, sends to ComfyUI on Windows LAN.

Usage:
    python batch_process.py --input frames/ --workflow ../workflows/controlnet_ipa.json
    python batch_process.py --input frames/ --workflow ../workflows/img2img.json --mode chained
"""

import argparse
import json
import time
from pathlib import Path

from comfyui_client import ComfyUIClient, load_workflow, parametrize_workflow

# Ported from BFL_FLUXdemos/src/experiments/emergent-worlds/prompts.ts
DEFAULT_PROMPT = (
    "simaesthetic, bioluminescent organic network, physarum slime mold veins "
    "pulsing with cyan and amber glow, scanning electron microscope aesthetic, "
    "extreme depth of field, dark background"
)

STRUCTURE_SUFFIX = (
    " Match the spatial distribution of detail from the input image exactly."
    " Areas with bright structure get rich detail and texture."
    " Areas that are dark or empty remain as flat, untextured, uniform surfaces"
    " with no added detail."
)

STYLE_REF_SUFFIX = " Maintain the rendering style and color palette from the style reference."


def build_prompt(base: str, has_style_ref: bool = False) -> str:
    """Build full prompt with structure enforcement suffix."""
    prompt = base + STRUCTURE_SUFFIX
    if has_style_ref:
        prompt += STYLE_REF_SUFFIX
    return prompt


def get_sorted_frames(input_dir: Path) -> list[Path]:
    """Get image files sorted by name."""
    exts = {".png", ".jpg", ".jpeg", ".webp"}
    frames = [f for f in input_dir.iterdir() if f.suffix.lower() in exts]
    return sorted(frames)


def process_parallel(
    client: ComfyUIClient,
    workflow_path: Path,
    frames: list[Path],
    output_dir: Path,
    prompt: str,
    params: dict,
) -> list[Path]:
    """Process all frames independently (parallel mode)."""
    results = []
    wf_base = load_workflow(workflow_path)
    full_prompt = build_prompt(prompt, has_style_ref=False)

    for i, frame in enumerate(frames):
        print(f"[{i+1}/{len(frames)}] Processing {frame.name}...")

        overrides = {"6.text": full_prompt, **params}
        wf = parametrize_workflow(wf_base, overrides)

        frame_out = output_dir / f"ai_{frame.stem}.png"
        saved = client.run_workflow(
            wf,
            images={"10.image": str(frame)},
            output_dir=output_dir,
            on_progress=lambda node, cur, total: print(f"  Step {cur}/{total}", end="\r"),
        )
        if saved:
            saved[0].rename(frame_out)
            results.append(frame_out)
            print(f"  -> {frame_out.name}")

    return results


def process_chained(
    client: ComfyUIClient,
    workflow_path: Path,
    frames: list[Path],
    output_dir: Path,
    prompt: str,
    params: dict,
) -> list[Path]:
    """Process frames sequentially, each using prev AI output as style ref (chained mode).

    Mirrors the 'chained' transform mode from EmergentWorlds.tsx:
    - Frame 1: sim capture -> FLUX -> AI1
    - Frame 2: AI1 (style ref) + sim capture -> FLUX -> AI2
    - Frame 3: AI2 (style ref) + sim capture -> FLUX -> AI3
    """
    results = []
    wf_base = load_workflow(workflow_path)
    prev_output: Path | None = None

    for i, frame in enumerate(frames):
        has_ref = prev_output is not None
        full_prompt = build_prompt(prompt, has_style_ref=has_ref)
        print(f"[{i+1}/{len(frames)}] Processing {frame.name}" + (" (with style ref)" if has_ref else ""))

        overrides = {"6.text": full_prompt, **params}
        wf = parametrize_workflow(wf_base, overrides)

        images = {"10.image": str(frame)}
        if has_ref:
            images["12.image"] = str(prev_output)

        frame_out = output_dir / f"ai_{frame.stem}.png"
        saved = client.run_workflow(
            wf,
            images=images,
            output_dir=output_dir,
            on_progress=lambda node, cur, total: print(f"  Step {cur}/{total}", end="\r"),
        )
        if saved:
            saved[0].rename(frame_out)
            results.append(frame_out)
            prev_output = frame_out
            print(f"  -> {frame_out.name}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Batch process DLA frames through ComfyUI")
    parser.add_argument("--input", "-i", required=True, help="Input frames directory")
    parser.add_argument("--workflow", "-w", required=True, help="Workflow JSON path")
    parser.add_argument("--output", "-o", default="outputs", help="Output directory")
    parser.add_argument("--host", default="http://127.0.0.1:8188", help="ComfyUI host")
    parser.add_argument("--mode", choices=["parallel", "chained"], default="parallel")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Base prompt")
    parser.add_argument("--denoise", type=float, default=0.6, help="Denoise strength")
    parser.add_argument("--steps", type=int, default=30, help="Sampling steps")
    parser.add_argument("--cfg", type=float, default=6.0, help="CFG scale")
    parser.add_argument("--seed", type=int, default=-1, help="Seed (-1 = random)")
    parser.add_argument("--limit", type=int, default=0, help="Max frames to process (0 = all)")
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    frames = get_sorted_frames(input_dir)
    if args.limit > 0:
        frames = frames[:args.limit]
    if not frames:
        print(f"No image files found in {input_dir}")
        return

    print(f"Found {len(frames)} frames")
    print(f"Mode: {args.mode}")
    print(f"Host: {args.host}")

    client = ComfyUIClient(args.host)
    params = {
        "3.seed": args.seed if args.seed >= 0 else int(time.time()),
        "3.steps": args.steps,
        "3.cfg": args.cfg,
        "3.denoise": args.denoise,
    }

    process_fn = process_chained if args.mode == "chained" else process_parallel
    start = time.time()
    results = process_fn(client, Path(args.workflow), frames, output_dir, args.prompt, params)
    elapsed = time.time() - start

    print(f"\nDone: {len(results)} images in {elapsed:.1f}s ({elapsed/len(results):.1f}s/img)")

    # Save session metadata
    meta = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "mode": args.mode,
        "prompt": args.prompt,
        "params": params,
        "input_frames": [f.name for f in frames],
        "output_files": [r.name for r in results],
        "elapsed_seconds": elapsed,
    }
    meta_path = output_dir / "session.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"Metadata: {meta_path}")


if __name__ == "__main__":
    main()
