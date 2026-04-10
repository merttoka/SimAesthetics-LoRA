"""Render overlay video of AI patches on solid background for compositing.

Generates a video where AI-rendered patches appear briefly (staggered pop + fade out)
at their original spatial positions from the simulation timeline. Supports dual AI
sources (e.g. SDXL early, FLUX late) with stochastic crossfade.

Output covers the central 30% of the ultrawide frame (4300x1920) so it can be
manually placed over the wider simulation video in Premiere.

Usage:
    # Test first 20 seconds
    python render_overlay_video.py \
        --manifest datasets/sim_aesthetic_2/manifest.json \
        --ai-dir outputs/flux_batch_v2/ \
        --out outputs/overlay_video_test.mp4 --test

    # Dual source: FLUX early → SDXL late
    python render_overlay_video.py \
        --manifest datasets/sim_aesthetic_2/manifest.json \
        --ai-dir-early outputs/flux_batch_v2/ \
        --ai-dir-late outputs/img2img-lora_v2-3/ \
        --crossfade-center 0.5 \
        --out outputs/overlay_video.mp4

    # Full render with gray background
    python render_overlay_video.py \
        --manifest datasets/sim_aesthetic_2/manifest.json \
        --ai-dir outputs/flux_batch_v2/ \
        --bg 128 \
        --out outputs/overlay_video.mp4
"""

import argparse
import json
import random
import re
import subprocess
import time
from dataclasses import dataclass, field, asdict
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image


# Source ultrawide dimensions
SOURCE_W = 14336
SOURCE_H = 1920

# Default output: central 30% of ultrawide
DEFAULT_OUT_W = 4300  # int(14336 * 0.3) ≈ 4300
DEFAULT_OUT_H = 1920

TOTAL_SIM_FRAMES = 45000
FPS = 60


def compute_layout(out_w, out_h):
    """Compute source window and scale factor for target output resolution.

    Returns (window_x_start, window_w_source, window_h_source, scale_x, scale_y).
    The source window matches the output aspect ratio, centered horizontally.
    """
    aspect = out_w / out_h
    # Source window: full height, width computed from aspect ratio
    window_h = SOURCE_H
    window_w = int(SOURCE_H * aspect)
    window_w = min(window_w, SOURCE_W)  # clamp to source width
    # Ensure even (h264)
    window_w = window_w - (window_w % 2)
    window_x_start = (SOURCE_W - window_w) // 2
    scale_x = out_w / window_w
    scale_y = out_h / window_h
    return window_x_start, window_w, window_h, scale_x, scale_y


@dataclass
class PatchEvent:
    ai_index: str
    ai_dir: str  # which AI source directory to pull from
    # Sub-crop within the 1024x1024 AI image
    src_x: int
    src_y: int
    src_size: int
    # Position in output canvas
    dst_x: int
    dst_y: int
    dst_size: int
    # Timeline (frame numbers, 0-indexed)
    frame_start: int
    frame_hold_end: int
    frame_fade_end: int


def load_manifest(path: Path) -> list[dict]:
    data = json.loads(path.read_text())
    return data["entries"]


def extract_sim_step(source_name: str) -> int:
    """Extract simulation step from filename like 'Image Sequence_002_0500.jpg'."""
    nums = re.findall(r'\d+', source_name)
    return int(nums[-1]) if nums else 0


def group_by_source(entries: list[dict]) -> dict[str, list[dict]]:
    groups = {}
    for entry in entries:
        src = entry["source"]
        if src not in groups:
            groups[src] = []
        groups[src].append(entry)
    return groups


def sample_patches(crop_size, n_patches, min_patch, max_patch, rng):
    """Sample non-overlapping square patches within a crop_size region."""
    patches = []
    attempts = 0
    while len(patches) < n_patches and attempts < n_patches * 50:
        attempts += 1
        size = rng.randint(min_patch, max_patch)
        x = rng.randint(0, crop_size - size)
        y = rng.randint(0, crop_size - size)
        overlaps = False
        for px, py, ps in patches:
            if x < px + ps and x + size > px and y < py + ps and y + size > py:
                overlaps = True
                break
        if not overlaps:
            patches.append((x, y, size))
    return patches


def compute_hold(sim_step, dst_size, max_patch, rng):
    """Hold duration: larger patches fade faster, smaller linger.

    Small patches hold up to ~1.5s (90 frames).
    Large patches hold ~0.15s (9 frames) — flash and fade.
    """
    size_t = min(1.0, dst_size / max(max_patch, 1))
    density_t = sim_step / TOTAL_SIM_FRAMES
    density_bonus = 1.0 - 0.3 * density_t

    # Inverted: small = long hold, large = short hold
    base = int((90 - 75 * size_t) * density_bonus)  # 90 (small) to 15 (large)
    return max(6, base + rng.randint(-3, 3))


def pick_ai_dir(sim_step, ai_dir_early, ai_dir_late, crossfade_center, crossfade_width, rng):
    """Stochastically pick early or late AI source based on simulation progress.

    Uses a sigmoid-like probability curve centered at crossfade_center.
    Early in the sim → favors ai_dir_early.
    Late in the sim → favors ai_dir_late.
    """
    if ai_dir_early is None:
        return ai_dir_late
    if ai_dir_late is None:
        return ai_dir_early

    t = sim_step / TOTAL_SIM_FRAMES  # 0.0 to 1.0
    # Sigmoid: probability of choosing "late" source
    # crossfade_width controls sharpness (smaller = sharper transition)
    x = (t - crossfade_center) / crossfade_width
    x = max(-10, min(10, x))  # clamp to avoid overflow
    p_late = 1.0 / (1.0 + np.exp(-x))

    return ai_dir_late if rng.random() < p_late else ai_dir_early


def generate_events(entries, ai_dir_early, ai_dir_late, crossfade_center, crossfade_width,
                    out_w, out_h, window_x_start, scale_x, scale_y,
                    seed=42, n_variations=4, min_patch=128, max_patch=512):
    """Generate all PatchEvents from manifest entries."""
    groups = group_by_source(entries)
    events = []
    rng = random.Random(seed)
    spread = 90  # ±90 frames = ~3s at 60fps

    source_counts = {"early": 0, "late": 0}
    # Max total event span: 4 seconds = 240 frames
    # Spread window: -150 to +90 frames from sample point (leads by ~2.5s, trails ~1.5s)
    spread_before = 150  # frames before sample point
    spread_after = 90    # frames after sample point

    for source_name, crops in sorted(groups.items()):
        sim_step = extract_sim_step(source_name)
        center_frame = sim_step

        source_patches = []

        for entry in crops:
            crop_size = entry["crop_size"]
            scale = crop_size / 1024.0

            n_var = rng.randint(n_variations, n_variations + 1)
            ai_min = max(int(min_patch / scale), 32)
            ai_max = min(int(max_patch / scale), 1024)

            patches = sample_patches(1024, n_var, ai_min, ai_max, rng)

            for px, py, ps in patches:
                src_x_in_source = entry["x"] + int(px * scale)
                src_y_in_source = entry["y"] + int(py * scale)
                dst_size_raw = int(ps * scale)

                # Map to output coordinates
                dst_x = int((src_x_in_source - window_x_start) * scale_x)
                dst_y = int(src_y_in_source * scale_y)
                dst_size = int(dst_size_raw * scale_x)  # use scale_x (≈scale_y)

                if dst_x + dst_size <= 0 or dst_x >= out_w:
                    continue
                if dst_y + dst_size <= 0 or dst_y >= out_h:
                    continue

                # Pick AI source stochastically
                chosen_dir = pick_ai_dir(
                    sim_step, ai_dir_early, ai_dir_late,
                    crossfade_center, crossfade_width, rng
                )
                if chosen_dir == ai_dir_early:
                    source_counts["early"] += 1
                else:
                    source_counts["late"] += 1

                source_patches.append((
                    entry["index"], str(chosen_dir),
                    px, py, ps, dst_x, dst_y, dst_size
                ))

        # Stagger patches across the asymmetric spread window
        rng.shuffle(source_patches)
        n_patches = len(source_patches)
        total_spread = spread_before + spread_after
        for i, (ai_idx, ai_dir, px, py, ps, dst_x, dst_y, dst_size) in enumerate(source_patches):
            # Distribute from -spread_before to +spread_after
            offset = int(-spread_before + (total_spread * i / max(n_patches - 1, 1)))
            stagger = rng.randint(0, 3)
            frame_start = center_frame + offset + stagger

            hold = compute_hold(sim_step, dst_size, max_patch, rng)
            fade = rng.randint(4, 10)

            events.append(PatchEvent(
                ai_index=ai_idx,
                ai_dir=ai_dir,
                src_x=px, src_y=py, src_size=ps,
                dst_x=dst_x, dst_y=dst_y, dst_size=dst_size,
                frame_start=max(0, frame_start),
                frame_hold_end=max(0, frame_start + hold),
                frame_fade_end=max(0, frame_start + hold + fade),
            ))

    events.sort(key=lambda e: e.frame_start)

    if ai_dir_early and ai_dir_late:
        total = source_counts["early"] + source_counts["late"]
        print(f"Source mix: {source_counts['early']} early ({source_counts['early']*100//total}%), "
              f"{source_counts['late']} late ({source_counts['late']*100//total}%)")

    return events


# Cache AI images (1024x1024x3 ≈ 3MB each, cache 30 = ~90MB)
@lru_cache(maxsize=30)
def load_ai_image(ai_dir: str, index: str) -> np.ndarray:
    path = Path(ai_dir) / f"ai_img_{index}.png"
    return np.array(Image.open(path).convert("RGB"))


# Cache resized patches to avoid redundant resizes across frames
@lru_cache(maxsize=100)
def get_resized_patch(ai_dir: str, ai_index: str, src_x: int, src_y: int,
                      src_size: int, dst_size: int) -> np.ndarray:
    ai_arr = load_ai_image(ai_dir, ai_index)
    patch = ai_arr[src_y:src_y + src_size, src_x:src_x + src_size]
    resized = Image.fromarray(patch).resize((dst_size, dst_size), Image.LANCZOS)
    return np.array(resized)


BORDER_COLOR = (204, 204, 204)  # #ccc
BORDER_WIDTH = 3
LABEL_COLOR = (204, 204, 204)  # #ccc


def get_label_font():
    """Get a small font for annotations."""
    from PIL import ImageFont
    for name in ["/System/Library/Fonts/Menlo.ttc", "/System/Library/Fonts/Helvetica.ttc", "arial.ttf"]:
        try:
            return ImageFont.truetype(name, 22)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


_label_font = None


def get_font():
    global _label_font
    if _label_font is None:
        _label_font = get_label_font()
    return _label_font


def make_event_label(event):
    """Build annotation string: source.res.batch"""
    source = "sd" if ("lora_v2" in event.ai_dir or "sdxl" in event.ai_dir.lower()) else "fl"
    res = f"{event.dst_size}x{event.dst_size}"
    idx = event.ai_index
    return f"{source} . {res} . b{idx}"


def draw_annotation(pil_img, dx, dy, size, label, alpha, bg_color, out_w, out_h):
    """Draw border + text label below the patch on a PIL Image."""
    from PIL import ImageDraw

    draw = ImageDraw.Draw(pil_img)
    font = get_font()

    # Fade color toward bg
    def fade_color(c):
        return tuple(int(c[i] * alpha + bg_color[i] * (1 - alpha)) for i in range(3))

    border_c = fade_color(BORDER_COLOR)
    label_c = fade_color(LABEL_COLOR)

    # Border rectangle
    bw = BORDER_WIDTH
    x0, y0 = dx, dy
    x1, y1 = dx + size - 1, dy + size - 1
    for i in range(bw):
        draw.rectangle([x0 + i, y0 + i, x1 - i, y1 - i], outline=border_c)

    # Text label below bottom-left of border
    text_y = y1 + 3
    text_x = x0
    if 0 <= text_x < out_w and 0 <= text_y < out_h - 10:
        draw.text((text_x, text_y), label, fill=label_c, font=font)


def render_video(events, output_path, total_frames, bg_color, out_w, out_h,
                 layer="patches", fps=60):
    """Render overlay video by streaming frames to ffmpeg.

    layer: "patches" (AI images only), "annotations" (borders + text only),
           or "composite" (patches + annotations combined)
    """
    proc = subprocess.Popen([
        "ffmpeg", "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{out_w}x{out_h}", "-r", str(fps),
        "-i", "-",
        "-c:v", "libx264", "-preset", "medium", "-crf", "18",
        "-pix_fmt", "yuv420p",
        str(output_path),
    ], stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)

    bg_r, bg_g, bg_b = bg_color
    bg_pixel = bytes([bg_r, bg_g, bg_b])
    bg_bytes = bg_pixel * (out_w * out_h)
    bg_array = np.array(bg_color, dtype=np.uint8)

    event_idx = 0
    active = []
    t0 = time.time()
    frames_with_patches = 0

    for frame in range(total_frames):
        while event_idx < len(events) and events[event_idx].frame_start <= frame:
            active.append(events[event_idx])
            event_idx += 1

        active = [e for e in active if frame <= e.frame_fade_end]

        if not active:
            proc.stdin.write(bg_bytes)
        else:
            frames_with_patches += 1

            if layer in ("patches", "composite"):
                canvas = np.empty((out_h, out_w, 3), dtype=np.uint8)
                canvas[:] = bg_array

                for e in active:
                    if frame <= e.frame_hold_end:
                        alpha = 1.0
                    else:
                        progress = (frame - e.frame_hold_end) / max(1, e.frame_fade_end - e.frame_hold_end)
                        alpha = 1.0 - progress

                    patch_arr = get_resized_patch(
                        e.ai_dir, e.ai_index,
                        e.src_x, e.src_y, e.src_size, e.dst_size
                    )

                    if alpha < 1.0:
                        patch_arr = (patch_arr.astype(np.float32) * alpha +
                                     bg_array.astype(np.float32) * (1.0 - alpha)).astype(np.uint8)

                    dx, dy = e.dst_x, e.dst_y
                    pw = ph = e.dst_size
                    sx0 = max(0, -dx)
                    sy0 = max(0, -dy)
                    dx0 = max(0, dx)
                    dy0 = max(0, dy)
                    sx1 = min(pw, out_w - dx)
                    sy1 = min(ph, out_h - dy)

                    if sx1 > sx0 and sy1 > sy0:
                        canvas[dy0:dy0 + (sy1 - sy0), dx0:dx0 + (sx1 - sx0)] = \
                            patch_arr[sy0:sy1, sx0:sx1]

                if layer == "composite":
                    # Overlay annotations on top of patches
                    pil_img = Image.fromarray(canvas)
                    for e in active:
                        if frame <= e.frame_hold_end:
                            alpha = 1.0
                        else:
                            progress = (frame - e.frame_hold_end) / max(1, e.frame_fade_end - e.frame_hold_end)
                            alpha = 1.0 - progress
                        label = make_event_label(e)
                        draw_annotation(pil_img, e.dst_x, e.dst_y, e.dst_size,
                                        label, alpha, bg_color, out_w, out_h)
                    proc.stdin.write(np.array(pil_img).tobytes())
                else:
                    proc.stdin.write(canvas.tobytes())

            elif layer == "annotations":
                # Use PIL for text rendering
                pil_img = Image.new("RGB", (out_w, out_h), bg_color)

                for e in active:
                    if frame <= e.frame_hold_end:
                        alpha = 1.0
                    else:
                        progress = (frame - e.frame_hold_end) / max(1, e.frame_fade_end - e.frame_hold_end)
                        alpha = 1.0 - progress

                    label = make_event_label(e)
                    draw_annotation(pil_img, e.dst_x, e.dst_y, e.dst_size,
                                    label, alpha, bg_color, out_w, out_h)

                proc.stdin.write(np.array(pil_img).tobytes())

        if (frame + 1) % 600 == 0 or frame == total_frames - 1:
            elapsed = time.time() - t0
            fps_actual = (frame + 1) / elapsed
            eta = (total_frames - frame - 1) / fps_actual if fps_actual > 0 else 0
            print(f"  frame {frame + 1}/{total_frames} "
                  f"({fps_actual:.1f} fps, ETA {eta:.0f}s, "
                  f"{frames_with_patches} active frames)")

    proc.stdin.close()
    proc.wait()

    if proc.returncode != 0:
        print(f"ffmpeg exited with code {proc.returncode}")
        return False

    elapsed = time.time() - t0
    print(f"\nDone: {output_path} ({elapsed:.0f}s)")
    return True


def main():
    parser = argparse.ArgumentParser(description="Render overlay video for compositing")
    parser.add_argument("--manifest", "-m", required=True, help="manifest.json path")
    parser.add_argument("--ai-dir", "-a", help="Single AI source directory (use for all frames)")
    parser.add_argument("--ai-dir-early", help="AI source for early frames (e.g. FLUX)")
    parser.add_argument("--ai-dir-late", help="AI source for late frames (e.g. SDXL)")
    parser.add_argument("--crossfade-center", type=float, default=0.5,
                        help="Timeline position (0-1) where early/late sources are 50/50 (default: 0.5)")
    parser.add_argument("--crossfade-width", type=float, default=0.15,
                        help="Crossfade sigmoid width — smaller = sharper transition (default: 0.15)")
    parser.add_argument("--bg", default="0,255,0",
                        help="Background color as R,G,B (default: 0,255,0 green screen)")
    parser.add_argument("--out", "-o", default="outputs/overlay_video.mp4", help="Output video path")
    parser.add_argument("--test", action="store_true", help="Render only first 20s (1200 frames)")
    parser.add_argument("--test-duration", type=int, default=20, help="Test duration in seconds (default: 20)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--variations", type=int, default=4, help="Variations per crop (4-5)")
    parser.add_argument("--min-patch", type=int, default=128, help="Min patch size in source pixels")
    parser.add_argument("--max-patch", type=int, default=512, help="Max patch size in source pixels")
    parser.add_argument("--layer", choices=["patches", "annotations", "composite"], default="patches",
                        help="Render layer: patches, annotations, or composite (both)")
    parser.add_argument("--resolution", default=None,
                        help="Output resolution as WxH (default: 4300x1920)")
    args = parser.parse_args()

    # Resolve output resolution
    if args.resolution:
        out_w, out_h = (int(x) for x in args.resolution.split("x"))
    else:
        out_w, out_h = DEFAULT_OUT_W, DEFAULT_OUT_H

    # Compute source window and scale
    window_x_start, window_w, window_h, scale_x, scale_y = compute_layout(out_w, out_h)

    # Resolve AI directories
    if args.ai_dir:
        ai_dir_early = None
        ai_dir_late = Path(args.ai_dir)
    elif args.ai_dir_early and args.ai_dir_late:
        ai_dir_early = Path(args.ai_dir_early)
        ai_dir_late = Path(args.ai_dir_late)
    else:
        parser.error("Provide --ai-dir or both --ai-dir-early and --ai-dir-late")
        return

    entries = load_manifest(Path(args.manifest))
    print(f"Manifest: {len(entries)} crops")
    if ai_dir_early:
        print(f"Early source: {ai_dir_early}")
        print(f"Late source:  {ai_dir_late}")
        print(f"Crossfade: center={args.crossfade_center}, width={args.crossfade_width}")
    else:
        print(f"AI source: {ai_dir_late}")
    bg_color = tuple(int(x) for x in args.bg.split(","))
    print(f"Background: RGB{bg_color}")
    print(f"Resolution: {out_w}x{out_h} (source window: {window_w}x{window_h} @ x={window_x_start})")

    events = generate_events(
        entries, ai_dir_early, ai_dir_late,
        args.crossfade_center, args.crossfade_width,
        out_w, out_h, window_x_start, scale_x, scale_y,
        seed=args.seed, n_variations=args.variations,
        min_patch=args.min_patch, max_patch=args.max_patch,
    )
    print(f"Generated {len(events)} patch events")

    total_frames = args.test_duration * FPS if args.test else TOTAL_SIM_FRAMES
    if args.test:
        events = [e for e in events if e.frame_start < total_frames + 60]
        print(f"Test mode: {total_frames} frames ({total_frames / FPS:.1f}s), "
              f"{len(events)} events in range")

    duration = total_frames / FPS
    print(f"Rendering {total_frames} frames at {FPS}fps ({duration:.1f}s), {out_w}x{out_h}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    # Save events for audio synthesis
    events_path = Path(args.out).with_suffix(".events.json")
    events_data = {
        "fps": FPS,
        "total_frames": total_frames,
        "out_w": out_w,
        "out_h": out_h,
        "events": [asdict(e) for e in events],
    }
    events_path.write_text(json.dumps(events_data))
    print(f"Saved {len(events)} events to {events_path}")

    print(f"Layer: {args.layer}")
    render_video(events, args.out, total_frames, bg_color, out_w, out_h, args.layer, FPS)


if __name__ == "__main__":
    main()
