"""Export DLA frames from the web app for ComfyUI processing.

Two export methods:
1. Screenshot mode: Playwright captures canvas from running web app
2. ZIP import: Extract frames from existing EmergentWorlds ZIP exports

Usage:
    # From running web app
    python export_frames.py screenshot --url http://localhost:5173/flux-reimagined-ecosystems --count 10 --interval 2

    # From existing ZIP export
    python export_frames.py unzip --input ew-export-*.zip --output frames/
"""

import argparse
import io
import json
import zipfile
from pathlib import Path

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


def extract_from_zip(zip_path: Path, output_dir: Path, prefix: str = "capture") -> list[Path]:
    """Extract capture frames from EmergentWorlds ZIP export.

    ZIP structure (from EmergentWorlds.tsx doExport):
        capture-001.png, capture-002.png, ...
        ai-001.png, ai-002.png, ...
        session.json
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    extracted = []

    with zipfile.ZipFile(zip_path) as zf:
        # Get capture files (sim frames)
        captures = sorted(n for n in zf.namelist() if n.startswith(f"{prefix}-"))

        for name in captures:
            data = zf.read(name)
            out_path = output_dir / name
            out_path.write_bytes(data)
            extracted.append(out_path)
            print(f"  Extracted: {name}")

        # Also extract session metadata if present
        if "session.json" in zf.namelist():
            session = json.loads(zf.read("session.json"))
            (output_dir / "session.json").write_text(json.dumps(session, indent=2))
            print(f"  Session metadata: {len(session.get('captures', []))} captures")

    print(f"Extracted {len(extracted)} frames to {output_dir}")
    return extracted


def extract_ai_frames(zip_path: Path, output_dir: Path) -> list[Path]:
    """Extract AI-transformed frames from ZIP (for comparison/dataset use)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    extracted = []

    with zipfile.ZipFile(zip_path) as zf:
        ai_frames = sorted(n for n in zf.namelist() if n.startswith("ai-"))
        for name in ai_frames:
            data = zf.read(name)
            out_path = output_dir / name
            out_path.write_bytes(data)
            extracted.append(out_path)

    print(f"Extracted {len(extracted)} AI frames to {output_dir}")
    return extracted


async def screenshot_frames(
    url: str,
    output_dir: Path,
    count: int = 10,
    interval: float = 2.0,
    resolution: int = 1024,
) -> list[Path]:
    """Capture DLA simulation frames via Playwright.

    Requires: pip install playwright && playwright install chromium
    """
    try:
        from playwright.async_api import async_playwright
    except ImportError:
        raise ImportError("pip install playwright && playwright install chromium")

    output_dir.mkdir(parents=True, exist_ok=True)
    frames = []

    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page(viewport={"width": resolution + 200, "height": resolution + 400})
        await page.goto(url)

        # Wait for WebGPU canvas
        await page.wait_for_selector("canvas", timeout=15000)
        print(f"Canvas found, capturing {count} frames at {interval}s intervals...")

        import asyncio
        for i in range(count):
            await asyncio.sleep(interval)

            # Capture the canvas element
            canvas = page.locator("canvas").first
            screenshot = await canvas.screenshot(type="png")

            frame_path = output_dir / f"capture-{i+1:03d}.png"
            frame_path.write_bytes(screenshot)
            frames.append(frame_path)
            print(f"  [{i+1}/{count}] {frame_path.name}")

        await browser.close()

    print(f"Captured {len(frames)} frames to {output_dir}")
    return frames


def main():
    parser = argparse.ArgumentParser(description="Export DLA frames")
    sub = parser.add_subparsers(dest="command", required=True)

    # ZIP extraction
    unzip = sub.add_parser("unzip", help="Extract frames from EmergentWorlds ZIP")
    unzip.add_argument("--input", "-i", required=True, help="ZIP file path")
    unzip.add_argument("--output", "-o", default="frames", help="Output directory")
    unzip.add_argument("--ai", action="store_true", help="Also extract AI frames")

    # Screenshot capture
    ss = sub.add_parser("screenshot", help="Capture frames from running web app")
    ss.add_argument("--url", default="http://localhost:5173/flux-reimagined-ecosystems")
    ss.add_argument("--output", "-o", default="frames", help="Output directory")
    ss.add_argument("--count", "-n", type=int, default=10, help="Number of frames")
    ss.add_argument("--interval", type=float, default=2.0, help="Seconds between captures")
    ss.add_argument("--resolution", type=int, default=1024, help="Canvas resolution")

    args = parser.parse_args()
    output_dir = Path(args.output)

    if args.command == "unzip":
        zip_path = Path(args.input)
        extract_from_zip(zip_path, output_dir)
        if args.ai:
            extract_ai_frames(zip_path, output_dir / "ai")

    elif args.command == "screenshot":
        import asyncio
        asyncio.run(screenshot_frames(
            args.url, output_dir, args.count, args.interval, args.resolution
        ))


if __name__ == "__main__":
    main()
