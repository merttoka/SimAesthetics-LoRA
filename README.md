# COMFY_SimAesthetics

ComfyUI + LoRA pipeline for generative simulation aesthetics. ALife simulations (Physarum, Boids from [Edge of Chaos](../UNITY_EoC_GPU)) → SDXL img2img + custom LoRA → photorealistic biological matter.

## Pipeline

```
Unity Edge of Chaos (Physarum + Boids)
└── 45k frames → every 500th → 91 ultrawide (14336x1920)
         │
    prepare_dataset.py
    (random 1024² crops, brightness filter, captions)
         │
         ▼
    270 training pairs → ai-toolkit LoRA (SDXL, rank 16)
    trigger: "simaesthetic" │ RunPod L40S ~55min
         │
         ▼
    ComfyUI (Windows 3080 LAN)
    ├── img2img: VAE Encode → KSampler (denoise 0.5-0.7)
    │   └── preserves void regions, adds texture to structure
    ├── ControlNet (Canny): structure-preserving generation
    └── LoRA strength 0.85-0.95: aesthetic intensity
         │
         ▼
    Output: sim skeleton → photorealistic biomass
    make_grid.py → side-by-side comparison grid
```

## Quick Start

```bash
# 1. Prepare dataset from sim frames (random 1024² crops)
python scripts/prepare_dataset.py -i recordings/ -o datasets/sim_aesthetic/ \
  -t biomaesthetic --type physarum --crop sample --samples 3 --h-focus 0.3 --size 1024

# 2. Train LoRA on RunPod (upload dataset + config)
./scripts/pod.sh upload datasets/sim_aesthetic /workspace/datasets/sim_aesthetic
./scripts/pod.sh upload scripts/train_config_sdxl_runpod.yaml /workspace/
# On pod: python run.py /workspace/train_config_sdxl_runpod.yaml
./scripts/pod.sh loras  # download checkpoints

# 3. Test checkpoints in ComfyUI txt2img (pick best step)
# Load LoRA in ComfyUI, try trigger "simaesthetic" + descriptive prompt

# 4. Generate outputs: img2img (denoise 0.5-0.7) or ControlNet
python scripts/batch_process.py -i datasets/sim_aesthetic/ -w workflows/sdxl_img2img.json

# 5. Build comparison grid
python scripts/make_grid.py --left datasets/sim_aesthetic/ --right outputs/ --out grid.png
```

## Structure

```
COMFY_SimAesthetics/
├── workflows/                # ComfyUI workflow JSONs (API format)
│   ├── sdxl_img2img.json
│   ├── sdxl_controlnet_canny.json
│   ├── sdxl_controlnet_ipa.json         # Full BFL API equivalent
│   ├── sdxl_controlnet_lora.json
│   └── flux_controlnet_depth_lora.json  # FLUX + Depth CN + LoRA (lifecycle prototype)
├── scripts/                  # Automation & tooling (runs on Mac)
│   ├── comfyui_client.py                # ComfyUI API client
│   ├── batch_process.py                 # Batch frame processing
│   ├── prepare_dataset.py              # LoRA dataset prep (random crops, brightness filter)
│   ├── make_grid.py                    # Side-by-side comparison grid builder
│   ├── pod.sh                          # RunPod SSH/SCP helper
│   ├── scrape_textures.py              # CC-licensed bio texture scraper
│   ├── raster_to_controlnet.py         # Satellite/GeoTIFF → ControlNet PNG
│   ├── export_frames.py                # DLA frame extraction
│   ├── sweep_denoise.py                # Parameter sweep comparisons
│   ├── sweep_txt2img.py                # KSampler settings sweep
│   ├── train_config_sdxl.yaml          # ai-toolkit SDXL config (local 3080)
│   ├── train_config_sdxl_runpod.yaml   # ai-toolkit SDXL config (RunPod L40S)
│   └── train_config_flux.yaml          # ai-toolkit Flux config (A100)
├── datasets/                 # Training datasets (img + txt pairs)
├── loras/                    # Trained LoRA checkpoints
├── outputs/                  # Generated outputs
└── docs/                     # Pipeline documentation, comparisons
```

## Key Findings

- **img2img > txt2img for sim frames**: VAE Encode + KSampler (denoise 0.5-0.7) preserves dark void regions. txt2img + ControlNet fills everything with content
- **Canny ControlNet is redundant**: sim frames ARE edges on black — Canny extraction → regeneration round-trips the same structure
- **LoRA trigger word needs high caption dropout**: `caption_dropout_rate: 0.05` → weak trigger. Need 0.10-0.15 for strong `simaesthetic` binding
- **Best settings**: LoRA 0.85-0.95, ControlNet 0.7-0.8 (Canny low=237/high=255), denoise 0.5-0.7, cfg 6.0

## Hardware

| Task | 3080 (10GB) | RunPod L40S | Mac |
|------|:-----------:|:----------:|:---:|
| SDXL img2img / ControlNet | 1024px | — | — |
| SDXL LoRA training (rank 16) | ~80hr | ~55min | — |
| Flux LoRA training | OOM | OOM (48GB) | — |
| Dataset prep / scripting | — | — | Yes |

## Models Required

**On Windows 3080:**
- `sd_xl_base_1.0.safetensors` + `sdxl_vae.safetensors`
- `control-lora-canny-rank256.safetensors` (SDXL ControlNet)
- Custom LoRA: `sim_aesthetic_sdxl.safetensors` (trained, rank 16)

**Custom nodes (install via ComfyUI Manager):**
ComfyUI-Manager, comfyui-controlnet-aux, ComfyUI_IPAdapter_plus
