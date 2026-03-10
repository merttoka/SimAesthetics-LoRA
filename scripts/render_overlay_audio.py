"""Synthesize sample-based audio synced to overlay video patch events.

Reads .events.json from render_overlay_video.py and triggers audio samples
mapped to patch properties. Also scatters bird samples as ambient boids texture.

Sample mapping:
  - glasstap: small patches (SDXL-favored)
  - tok: medium patches
  - glitch: large patches (FLUX-favored)
  - birds3: ambient layer scattered across entire timeline (boids)

Usage:
    python render_overlay_audio.py \
        --events outputs/overlay_video.events.json \
        --samples inputs/ \
        --out outputs/overlay_audio.wav
"""

import argparse
import json
import struct
import time
import wave
from pathlib import Path

import numpy as np


SAMPLE_RATE = 44100


def load_wav(path):
    """Load a WAV file as float32 mono array."""
    with wave.open(str(path), "r") as w:
        frames = w.readframes(w.getnframes())
        sr = w.getframerate()
        ch = w.getnchannels()
        sw = w.getsampwidth()

    if sw == 2:
        data = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32767.0
    elif sw == 1:
        data = np.frombuffer(frames, dtype=np.uint8).astype(np.float32) / 128.0 - 1.0
    else:
        data = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32767.0

    # Mix to mono if stereo
    if ch > 1:
        data = data.reshape(-1, ch).mean(axis=1)

    # Resample if needed (simple linear interpolation)
    if sr != SAMPLE_RATE:
        duration = len(data) / sr
        new_len = int(duration * SAMPLE_RATE)
        x_old = np.linspace(0, 1, len(data))
        x_new = np.linspace(0, 1, new_len)
        data = np.interp(x_new, x_old, data)

    return data


def load_sample_bank(base_dir):
    """Load all sample folders into a dict of {name: [array, ...]}."""
    base = Path(base_dir)
    bank = {}
    for folder in sorted(base.iterdir()):
        if not folder.is_dir():
            continue
        samples = []
        for f in sorted(folder.glob("*.wav")):
            samples.append(load_wav(f))
        if samples:
            bank[folder.name] = samples
            print(f"  {folder.name}: {len(samples)} samples")
    return bank


def pitch_shift(sample, factor):
    """Pitch shift by resampling. factor > 1 = higher pitch."""
    if abs(factor - 1.0) < 0.01:
        return sample
    new_len = int(len(sample) / factor)
    if new_len < 2:
        return sample
    x_old = np.linspace(0, 1, len(sample))
    x_new = np.linspace(0, 1, new_len)
    return np.interp(x_new, x_old, sample)


def apply_envelope(sample, attack_ms=15, decay_factor=1.0):
    """Apply soft attack and optional decay to a sample."""
    out = sample.copy()

    # Soft attack
    attack_samples = min(int(attack_ms / 1000 * SAMPLE_RATE), len(out) // 3)
    if attack_samples > 1:
        out[:attack_samples] *= 0.5 * (1.0 - np.cos(
            np.pi * np.linspace(0, 1, attack_samples)))

    # Exponential decay tail
    if decay_factor > 0:
        t = np.arange(len(out), dtype=np.float32) / SAMPLE_RATE
        out *= np.exp(-decay_factor * t)

    return out


def make_stereo(mono, pan):
    """Convert mono to stereo with constant-power panning. pan: -1 to +1."""
    pan_angle = (pan * 0.7 + 1.0) / 2.0  # slight narrowing
    left = mono * np.cos(pan_angle * np.pi / 2)
    right = mono * np.sin(pan_angle * np.pi / 2)
    return np.stack([left, right], axis=-1)


def mix_into(output, stereo_sample, sample_start, volume):
    """Mix a stereo sample into the output buffer at given position."""
    end_idx = min(sample_start + len(stereo_sample), len(output))
    actual_len = end_idx - sample_start
    if actual_len > 0:
        output[sample_start:end_idx] += stereo_sample[:actual_len] * volume


def apply_reverb(audio, decay=0.3, delay_ms=80):
    """Multi-tap comb filter reverb."""
    result = audio.copy()
    taps = [(delay_ms, decay), (delay_ms * 1.7, decay * 0.6),
            (delay_ms * 2.9, decay * 0.35), (delay_ms * 4.1, decay * 0.2)]
    for ms, d in taps:
        delay_samples = int(ms / 1000 * SAMPLE_RATE)
        if delay_samples < len(audio):
            result[delay_samples:] += audio[:-delay_samples] * d
    return result


def generate_bird_events(total_samples, rng, density=3.0):
    """Generate stochastic bird chirp positions across the full timeline.

    density: average chirps per second.
    Returns list of (sample_position, pan) tuples.
    """
    events = []
    duration_s = total_samples / SAMPLE_RATE
    n_chirps = int(duration_s * density)

    for _ in range(n_chirps):
        pos = rng.randint(0, total_samples)
        pan = rng.uniform(-1.0, 1.0)
        events.append((pos, pan))

    events.sort(key=lambda x: x[0])
    return events


def render_audio(events_data, sample_bank, output_path,
                 volume=0.03, bird_volume=0.02, reverb_amount=0.3):
    """Render sample-based audio from patch events + ambient birds."""
    fps = events_data["fps"]
    total_frames = events_data["total_frames"]
    canvas_w = events_data["out_w"]
    canvas_h = events_data["out_h"]
    events = events_data["events"]

    duration_s = total_frames / fps
    total_samples = int(duration_s * SAMPLE_RATE)

    print(f"Rendering {duration_s:.1f}s audio ({total_samples} samples)")

    max_patch = max(e["dst_size"] for e in events) if events else 800
    output = np.zeros((total_samples, 2), dtype=np.float32)
    rng = np.random.RandomState(42)

    # --- Patch event triggers ---
    birds = sample_bank.get("birds3", [])
    tok = sample_bank.get("tok", [])
    # glasstap used as "tap" — short delicate triggers
    tap = sample_bank.get("glasstap", [])

    # Combined pools: birds + tok + tap, weighted by patch size
    # Small patches → tap + birds, large → tok + birds
    print(f"  Rendering {len(events)} patch triggers...")
    for i, event in enumerate(events):
        sample_start = int(event["frame_start"] / fps * SAMPLE_RATE)
        if sample_start >= total_samples:
            continue

        size_t = event["dst_size"] / max(max_patch, 1)

        # Birds dominant (70%), tok for body (30%)
        r = rng.random()
        if r < 0.7 and birds:
            pool = birds
        else:
            pool = tok if tok else birds

        if not pool:
            continue

        sample = pool[rng.randint(0, len(pool))].copy()

        # Pitch shift based on size (larger = lower, ±30%)
        pitch_factor = 0.7 + 0.6 * (1.0 - size_t)  # 0.7x (large) to 1.3x (small)
        pitch_factor *= 1.0 + rng.uniform(-0.08, 0.08)  # random wobble
        sample = pitch_shift(sample, pitch_factor)

        # Decay: larger patches ring longer
        decay = 1.0 + 4.0 * (1.0 - size_t)  # large=slow decay, small=fast
        attack = rng.uniform(10, 35)
        sample = apply_envelope(sample, attack_ms=attack, decay_factor=decay)

        # Stereo position
        center_x = event["dst_x"] + event["dst_size"] / 2
        pan = (center_x / canvas_w) * 2.0 - 1.0

        stereo = make_stereo(sample, pan)
        mix_into(output, stereo, sample_start, volume)

        if (i + 1) % 300 == 0:
            print(f"    {i+1}/{len(events)}")

    # --- Reverb ---
    if reverb_amount > 0:
        print("  Applying reverb...")
        output = apply_reverb(output, decay=reverb_amount)

    # Soft clip
    peak = np.max(np.abs(output))
    if peak > 1.0:
        print(f"  Peak {peak:.2f}, soft clipping")
        output = np.tanh(output)

    # Normalize
    peak = np.max(np.abs(output))
    if peak > 0:
        output *= 0.9 / peak

    print(f"  Writing {output_path}...")
    write_wav(output_path, output, SAMPLE_RATE)

    elapsed_info = f"peak={np.max(np.abs(output)):.2f}"
    print(f"\nDone: {output_path} ({elapsed_info})")


def write_wav(path, audio, sample_rate):
    """Write stereo float32 audio to 16-bit WAV."""
    audio_int = (audio * 32767).astype(np.int16)
    n_samples, n_channels = audio_int.shape

    with open(str(path), "wb") as f:
        data_size = n_samples * n_channels * 2
        f.write(b"RIFF")
        f.write(struct.pack("<I", 36 + data_size))
        f.write(b"WAVE")
        f.write(b"fmt ")
        f.write(struct.pack("<I", 16))
        f.write(struct.pack("<H", 1))
        f.write(struct.pack("<H", n_channels))
        f.write(struct.pack("<I", sample_rate))
        f.write(struct.pack("<I", sample_rate * n_channels * 2))
        f.write(struct.pack("<H", n_channels * 2))
        f.write(struct.pack("<H", 16))
        f.write(b"data")
        f.write(struct.pack("<I", data_size))
        f.write(audio_int.tobytes())


def main():
    parser = argparse.ArgumentParser(description="Sample-based audio from overlay events")
    parser.add_argument("--events", "-e", required=True, help="Path to .events.json")
    parser.add_argument("--samples", "-s", default="inputs/", help="Sample bank directory")
    parser.add_argument("--out", "-o", default="outputs/overlay_audio.wav", help="Output WAV path")
    parser.add_argument("--volume", type=float, default=0.03, help="Patch trigger volume (default: 0.03)")
    parser.add_argument("--bird-volume", type=float, default=0.02, help="Bird ambient volume (default: 0.02)")
    parser.add_argument("--bird-density", type=float, default=3.0, help="Bird chirps per second (default: 3.0)")
    parser.add_argument("--reverb", type=float, default=0.3, help="Reverb amount 0-1 (default: 0.3)")
    args = parser.parse_args()

    events_data = json.loads(Path(args.events).read_text())
    print(f"Loaded {len(events_data['events'])} events, "
          f"{events_data['total_frames']} frames at {events_data['fps']}fps")

    print("Loading samples...")
    sample_bank = load_sample_bank(args.samples)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    render_audio(events_data, sample_bank, args.out,
                 args.volume, args.bird_volume, args.reverb)


if __name__ == "__main__":
    main()
