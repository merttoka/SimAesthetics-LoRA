"""Sweep KSampler settings on txt2img via ComfyUI API.

Usage:
    python sweep_txt2img.py --host http://127.0.0.1:8188
"""

import argparse
import json
import time
import copy
from pathlib import Path
from comfyui_client import ComfyUIClient

# Minimal txt2img workflow matching current ComfyUI setup
WORKFLOW = {
    "1": {
        "class_type": "CheckpointLoaderSimple",
        "inputs": {"ckpt_name": "sd_xl_base_1.0.safetensors"}
    },
    "2": {
        "class_type": "CLIPTextEncode",
        "inputs": {"text": "", "clip": ["1", 1]}
    },
    "3": {
        "class_type": "KSampler",
        "inputs": {
            "seed": 42,
            "steps": 20,
            "cfg": 8.0,
            "sampler_name": "euler",
            "scheduler": "normal",
            "denoise": 1.0,
            "model": ["1", 0],
            "positive": ["6", 0],
            "negative": ["2", 0],
            "latent_image": ["5", 0]
        }
    },
    "4": {
        "class_type": "VAEDecode",
        "inputs": {"samples": ["3", 0], "vae": ["1", 2]}
    },
    "5": {
        "class_type": "EmptyLatentImage",
        "inputs": {"width": 512, "height": 512, "batch_size": 1}
    },
    "6": {
        "class_type": "CLIPTextEncode",
        "inputs": {
            "text": "beautiful scenery nature glass bottle landscape, purple galaxy bottle,",
            "clip": ["1", 1]
        }
    },
    "9": {
        "class_type": "SaveImage",
        "inputs": {"filename_prefix": "sweep", "images": ["4", 0]}
    }
}

SWEEPS = {
    "sampler": {
        "param": "sampler_name",
        "values": ["euler", "dpmpp_2m", "dpmpp_2m_sde", "dpmpp_3m_sde"],
    },
    "cfg": {
        "param": "cfg",
        "values": [3.0, 5.0, 7.0, 9.0, 12.0],
    },
    "steps": {
        "param": "steps",
        "values": [10, 20, 30, 50],
    },
    "scheduler": {
        "param": "scheduler",
        "values": ["normal", "karras", "exponential", "sgm_uniform"],
    },
}


def run_sweep(client, sweep_name, seed):
    sweep = SWEEPS[sweep_name]
    param = sweep["param"]
    values = sweep["values"]

    print(f"\n=== Sweeping {param} ===")
    for val in values:
        wf = copy.deepcopy(WORKFLOW)
        wf["3"]["inputs"][param] = val
        wf["3"]["inputs"]["seed"] = seed
        wf["9"]["inputs"]["filename_prefix"] = f"sweep_{param}_{val}"

        print(f"  {param}={val} ...", end=" ", flush=True)
        start = time.time()
        prompt_id = client.queue_prompt(wf)

        # Poll until done
        while True:
            history = client.get_history(prompt_id)
            if prompt_id in history:
                elapsed = time.time() - start
                print(f"done ({elapsed:.1f}s)")
                break
            time.sleep(1)


def main():
    parser = argparse.ArgumentParser(description="Sweep txt2img settings")
    parser.add_argument("--host", default="http://127.0.0.1:8188")
    parser.add_argument("--sweep", choices=list(SWEEPS.keys()) + ["all"], default="all")
    parser.add_argument("--seed", type=int, default=42, help="Fixed seed for comparison")
    parser.add_argument("--prompt", help="Override positive prompt")
    args = parser.parse_args()

    client = ComfyUIClient(args.host)

    if args.prompt:
        WORKFLOW["6"]["inputs"]["text"] = args.prompt

    # Test connection
    try:
        q = client.get_queue()
        print(f"Connected to {args.host}")
    except Exception as e:
        print(f"Can't reach {args.host}: {e}")
        return

    sweeps = list(SWEEPS.keys()) if args.sweep == "all" else [args.sweep]
    for s in sweeps:
        run_sweep(client, s, args.seed)

    print("\nDone. Check ComfyUI output/ folder for results.")


if __name__ == "__main__":
    main()
