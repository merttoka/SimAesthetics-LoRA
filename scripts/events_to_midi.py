"""Export overlay events as a MIDI file for driving audio in a DAW.

Reads .events.json from render_overlay_video.py and generates a standard
MIDI file (.mid) that can be loaded in any DAW with PAM or other sampler.

Mapping:
  - Notes C1-B2 (24 notes): mapped from patch size (large=low, small=high)
  - MIDI channels 1-2: ch1 = SDXL source, ch2 = FLUX source
  - Velocity: inversely mapped from patch size (small=soft, large=loud)
  - Note duration: matches visual hold + fade timing
  - CC10 (Pan): mapped from x position on canvas
  - CC1 (Mod wheel): mapped from y position

Load in DAW → PAM cells:
  - Cells 1-4: bird samples (MIDI ch 1-2, notes C1-B1)
  - Cells 5-8: tok samples (MIDI ch 1-2, notes C2-B2)
  Or assign however you like — the MIDI is just triggers.

Usage:
    python events_to_midi.py \
        --events outputs/overlay_video.events.json \
        --out outputs/overlay_events.mid
"""

import argparse
import json
import struct
from pathlib import Path


def write_variable_length(value):
    """Encode an integer as MIDI variable-length quantity."""
    result = []
    result.append(value & 0x7F)
    value >>= 7
    while value > 0:
        result.append((value & 0x7F) | 0x80)
        value >>= 7
    result.reverse()
    return bytes(result)


def write_midi_file(path, events, ticks_per_beat=480, bpm=120.0):
    """Write a standard MIDI file (format 0) from event list."""
    # Microseconds per beat for tempo
    us_per_beat = int(60_000_000 / bpm)

    midi_events = []

    # Tempo event at tick 0
    midi_events.append((0, bytes([0xFF, 0x51, 0x03,
                                   (us_per_beat >> 16) & 0xFF,
                                   (us_per_beat >> 8) & 0xFF,
                                   us_per_beat & 0xFF])))

    for event in events:
        midi_events.extend(event)

    # Sort by tick, stable
    midi_events.sort(key=lambda x: x[0])

    # Build track data
    track_data = bytearray()
    last_tick = 0

    for tick, data in midi_events:
        delta = max(0, tick - last_tick)
        track_data.extend(write_variable_length(delta))
        track_data.extend(data)
        last_tick = tick

    # End of track
    track_data.extend(write_variable_length(0))
    track_data.extend(bytes([0xFF, 0x2F, 0x00]))

    # Write file
    with open(str(path), "wb") as f:
        # Header: format 0, 1 track
        f.write(b"MThd")
        f.write(struct.pack(">I", 6))
        f.write(struct.pack(">HHH", 0, 1, ticks_per_beat))

        # Track chunk
        f.write(b"MTrk")
        f.write(struct.pack(">I", len(track_data)))
        f.write(track_data)


def events_to_midi(events_data, output_path, bpm=120.0):
    """Convert overlay events to MIDI."""
    fps = events_data["fps"]
    canvas_w = events_data["out_w"]
    canvas_h = events_data["out_h"]
    events = events_data["events"]

    ticks_per_beat = 480
    # Seconds to ticks conversion
    beats_per_second = bpm / 60.0
    ticks_per_second = ticks_per_beat * beats_per_second

    max_patch = max(e["dst_size"] for e in events) if events else 800
    min_patch = min(e["dst_size"] for e in events) if events else 100

    midi_events = []
    note_range_low = 36   # C2
    note_range_high = 72  # C5
    note_span = note_range_high - note_range_low

    print(f"Converting {len(events)} events to MIDI")
    print(f"  BPM: {bpm}, ticks/beat: {ticks_per_beat}")
    print(f"  Patch size range: {min_patch}-{max_patch}")

    sdxl_count = 0
    flux_count = 0

    for event in events:
        frame_start = event["frame_start"]
        frame_end = event["frame_fade_end"]

        # Time in seconds, then to ticks
        t_start = frame_start / fps
        t_end = frame_end / fps
        tick_start = int(t_start * ticks_per_second)
        tick_end = int(t_end * ticks_per_second)
        duration_ticks = max(1, tick_end - tick_start)

        # Source → MIDI channel (0-indexed: 0 = ch1 SDXL, 1 = ch2 FLUX)
        is_sdxl = "lora_v2" in event["ai_dir"] or "sdxl" in event["ai_dir"].lower()
        channel = 0 if is_sdxl else 1
        if is_sdxl:
            sdxl_count += 1
        else:
            flux_count += 1

        # Patch size → note (large = low, small = high)
        size_t = (event["dst_size"] - min_patch) / max(max_patch - min_patch, 1)
        size_t = max(0.0, min(1.0, size_t))
        note = note_range_low + int((1.0 - size_t) * note_span)
        note = max(note_range_low, min(note_range_high, note))

        # Velocity: moderate range, larger patches slightly louder
        velocity = 40 + int(size_t * 60)  # 40-100

        # Pan CC (CC10): x position → 0-127
        center_x = event["dst_x"] + event["dst_size"] / 2
        pan = int(max(0, min(127, (center_x / canvas_w) * 127)))

        # Y position → mod wheel (CC1): 0-127
        y_norm = max(0, min(1, event["dst_y"] / canvas_h))
        mod = int(y_norm * 127)

        # CC events (pan + mod) just before note on
        cc_tick = max(0, tick_start - 1)

        event_list = [
            # Pan CC
            (cc_tick, bytes([0xB0 | channel, 10, pan])),
            # Mod wheel CC
            (cc_tick, bytes([0xB0 | channel, 1, mod])),
            # Note On
            (tick_start, bytes([0x90 | channel, note, velocity])),
            # Note Off
            (tick_start + duration_ticks, bytes([0x80 | channel, note, 0])),
        ]
        midi_events.append(event_list)

    print(f"  SDXL events (ch1): {sdxl_count}")
    print(f"  FLUX events (ch2): {flux_count}")

    # Flatten
    all_events = []
    for group in midi_events:
        all_events.extend(group)

    write_midi_file(output_path, [all_events], ticks_per_beat, bpm)
    print(f"\nSaved: {output_path}")
    print(f"  Duration: {events_data['total_frames'] / fps:.1f}s")
    print(f"  Notes: {note_range_low} (C2, large patches) to {note_range_high} (C5, small patches)")
    print(f"  Ch1 = SDXL source, Ch2 = FLUX source")


def main():
    parser = argparse.ArgumentParser(description="Export overlay events as MIDI")
    parser.add_argument("--events", "-e", required=True, help="Path to .events.json")
    parser.add_argument("--out", "-o", default="outputs/overlay_events.mid", help="Output MIDI path")
    parser.add_argument("--bpm", type=float, default=120.0, help="BPM for MIDI timing (default: 120)")
    args = parser.parse_args()

    events_data = json.loads(Path(args.events).read_text())
    print(f"Loaded {len(events_data['events'])} events, "
          f"{events_data['total_frames']} frames at {events_data['fps']}fps")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    events_to_midi(events_data, args.out, args.bpm)


if __name__ == "__main__":
    main()
