"""Scrape CC-licensed biological texture images for LoRA training.

Downloads from iNaturalist and Wikimedia Commons with proper attribution.
Targets ~50-100 high-res images of biological textures across lifecycle stages.

Usage:
    python scrape_textures.py --preset lifecycle_growth --output ../datasets/raw/biomass_growth/ --limit 30
    python scrape_textures.py --source inaturalist --query "Physarum polycephalum" --output ../datasets/raw/physarum/ --limit 20
    python scrape_textures.py --source wikimedia --query "SEM fungi" --output ../datasets/raw/sem/ --limit 20
    python scrape_textures.py --preset all --output ../datasets/raw/biomass/ --limit 100
"""

import argparse
import json
import time
import urllib.request
import urllib.error
import urllib.parse
from pathlib import Path


# ── Search presets ────────────────────────────────────────────

PRESETS = {
    "lifecycle_growth": {
        "inaturalist": [
            "Physarum polycephalum",
            "slime mold",
            "mycelium growth",
            "coral polyp",
        ],
    },
    "lifecycle_decay": {
        "inaturalist": [
            "decomposing fungus",
            "bracket fungus",
        ],
        "wikimedia": [
            "decomposing organic matter",
            "rot macro photography",
            "necrotic tissue",
        ],
    },
    "sem_textures": {
        "wikimedia": [
            "scanning electron microscope biological",
            "SEM fungi",
            "SEM bone",
            "SEM pollen",
        ],
    },
    "rust_patina": {
        "wikimedia": [
            "rust texture macro",
            "oxidation pattern",
            "corrosion close up",
            "patina metal",
        ],
    },
}


# ── API helpers ───────────────────────────────────────────────

def _fetch_json(url: str) -> dict:
    """Fetch JSON from URL with User-Agent header."""
    req = urllib.request.Request(url, headers={
        "User-Agent": "COMFY_SimAesthetics/1.0 (LoRA training dataset builder)",
    })
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode())


def _download_image(url: str, dest: Path, min_size: int = 0) -> bool:
    """Download image to dest. Returns False on failure or if too small."""
    req = urllib.request.Request(url, headers={
        "User-Agent": "COMFY_SimAesthetics/1.0",
    })
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            # Check Content-Length if available
            content_length = resp.headers.get("Content-Length")
            if content_length and min_size > 0:
                # Rough heuristic: 512x512 JPEG ~ 50KB minimum
                if int(content_length) < 50_000:
                    return False
            data = resp.read()
            dest.write_bytes(data)
            return True
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as e:
        print(f"  SKIP download error: {e}")
        return False


# ── iNaturalist ───────────────────────────────────────────────

def search_inaturalist(query: str, limit: int = 30) -> list[dict]:
    """Search iNaturalist for CC-licensed observation photos."""
    params = urllib.parse.urlencode({
        "taxon_name": query,
        "photo_license": "cc-by,cc-by-sa,cc0",
        "quality_grade": "research",
        "photos": "true",
        "per_page": min(limit, 30),
        "order_by": "votes",
    })
    url = f"https://api.inaturalist.org/v1/observations?{params}"

    try:
        data = _fetch_json(url)
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
        print(f"  iNaturalist API error: {e}")
        return []

    results = []
    for obs in data.get("results", []):
        for photo in obs.get("photos", []):
            if len(results) >= limit:
                break
            photo_url = photo.get("url", "")
            if not photo_url:
                continue
            # Replace "square" with "large" for higher res
            large_url = photo_url.replace("/square.", "/large.")
            license_code = photo.get("license_code", "unknown")
            attribution = photo.get("attribution", "")
            results.append({
                "url": large_url,
                "license": license_code or "unknown",
                "attribution": attribution,
                "source": "inaturalist",
                "search_term": query,
                "observation_id": obs.get("id"),
            })
    return results


# ── Wikimedia Commons ─────────────────────────────────────────

def search_wikimedia(query: str, limit: int = 30) -> list[dict]:
    """Search Wikimedia Commons for images."""
    params = urllib.parse.urlencode({
        "action": "query",
        "generator": "search",
        "gsrsearch": query,
        "gsrnamespace": "6",
        "gsrlimit": min(limit, 50),
        "prop": "imageinfo",
        "iiprop": "url|extmetadata|size",
        "iiurlwidth": "1024",
        "format": "json",
    })
    url = f"https://commons.wikimedia.org/w/api.php?{params}"

    try:
        data = _fetch_json(url)
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
        print(f"  Wikimedia API error: {e}")
        return []

    pages = data.get("query", {}).get("pages", {})
    results = []
    for page in pages.values():
        if len(results) >= limit:
            break
        imageinfo = page.get("imageinfo", [{}])
        if not imageinfo:
            continue
        info = imageinfo[0]

        # Skip small images
        width = info.get("width", 0)
        height = info.get("height", 0)
        if width < 512 or height < 512:
            continue

        # Prefer thumburl (resized to 1024), fall back to full url
        img_url = info.get("thumburl") or info.get("url", "")
        if not img_url:
            continue

        meta = info.get("extmetadata", {})
        license_name = meta.get("LicenseShortName", {}).get("value", "unknown")
        artist = meta.get("Artist", {}).get("value", "unknown")

        results.append({
            "url": img_url,
            "license": license_name,
            "attribution": artist,
            "source": "wikimedia",
            "search_term": query,
            "title": page.get("title", ""),
            "width": width,
            "height": height,
        })
    return results


# ── Download engine ───────────────────────────────────────────

def download_results(
    results: list[dict],
    output_dir: Path,
    limit: int = 100,
) -> list[dict]:
    """Download images from search results with rate limiting."""
    output_dir.mkdir(parents=True, exist_ok=True)
    downloaded = []
    seen_urls = set()

    for i, entry in enumerate(results):
        if len(downloaded) >= limit:
            break

        url = entry["url"]
        if url in seen_urls:
            continue
        seen_urls.add(url)

        # Determine extension from URL
        ext = ".jpg"
        url_path = urllib.parse.urlparse(url).path.lower()
        for candidate in (".png", ".jpeg", ".jpg", ".webp", ".tiff"):
            if candidate in url_path:
                ext = candidate
                break

        filename = f"{len(downloaded)+1:03d}_{entry['source']}_{entry['search_term'][:20].replace(' ', '_')}{ext}"
        dest = output_dir / filename

        print(f"  [{len(downloaded)+1}/{limit}] {url[:80]}...")
        if _download_image(url, dest):
            entry["local_file"] = filename
            downloaded.append(entry)
        else:
            if dest.exists():
                dest.unlink()

        # Rate limit: 1 second between requests
        time.sleep(1)

    return downloaded


# ── Main ──────────────────────────────────────────────────────

def run_preset(preset_name: str, output_dir: Path, limit: int):
    """Run a curated search preset."""
    if preset_name == "all":
        presets = list(PRESETS.keys())
    else:
        presets = [preset_name]

    per_preset_limit = max(limit // len(presets), 10)
    all_results = []

    for name in presets:
        preset = PRESETS[name]
        print(f"\n── Preset: {name} ──")

        for source, queries in preset.items():
            per_query_limit = max(per_preset_limit // len(queries), 5)
            for query in queries:
                print(f"  Searching {source}: {query}")
                if source == "inaturalist":
                    results = search_inaturalist(query, per_query_limit)
                elif source == "wikimedia":
                    results = search_wikimedia(query, per_query_limit)
                else:
                    continue
                print(f"    Found {len(results)} results")
                all_results.extend(results)
                time.sleep(1)  # Rate limit between searches

    print(f"\nTotal candidates: {len(all_results)}")
    downloaded = download_results(all_results, output_dir, limit)
    return downloaded


def run_single(source: str, query: str, output_dir: Path, limit: int):
    """Run a single source query."""
    print(f"Searching {source}: {query}")
    if source == "inaturalist":
        results = search_inaturalist(query, limit)
    elif source == "wikimedia":
        results = search_wikimedia(query, limit)
    else:
        print(f"Unknown source: {source}")
        return []

    print(f"Found {len(results)} results")
    downloaded = download_results(results, output_dir, limit)
    return downloaded


def main():
    parser = argparse.ArgumentParser(description="Scrape CC-licensed biological textures")
    parser.add_argument("--source", "-s", choices=["inaturalist", "wikimedia"],
                        help="Search source (use with --query)")
    parser.add_argument("--query", "-q", help="Search query (use with --source)")
    parser.add_argument("--preset", "-p", choices=list(PRESETS.keys()) + ["all"],
                        help="Curated search preset")
    parser.add_argument("--output", "-o", required=True, help="Output directory")
    parser.add_argument("--limit", "-l", type=int, default=50, help="Max images to download")
    args = parser.parse_args()

    if not args.preset and not (args.source and args.query):
        parser.error("Provide --preset OR --source + --query")

    output_dir = Path(args.output)

    if args.preset:
        downloaded = run_preset(args.preset, output_dir, args.limit)
    else:
        downloaded = run_single(args.source, args.query, output_dir, args.limit)

    # Save metadata
    if downloaded:
        meta_path = output_dir / "metadata.json"
        meta_path.write_text(json.dumps({
            "total_downloaded": len(downloaded),
            "images": downloaded,
        }, indent=2))
        print(f"\nDone: {len(downloaded)} images saved to {output_dir}")
        print(f"Metadata: {meta_path}")
    else:
        print("\nNo images downloaded.")


if __name__ == "__main__":
    main()
