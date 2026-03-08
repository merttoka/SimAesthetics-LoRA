# COMFY_SimAesthetics

ComfyUI + LoRA pipeline for generative simulation aesthetics. ALife simulations (Physarum, Boids, CCA from [Edge of Chaos](../edge-of-chaos-unity-compute)) → structural masks → FLUX ControlNet + custom LoRA → photorealistic biological matter.

## Pipeline

```
ALife Simulations (Unity GPU)         Biological Texture Sources
├── Physarum (growth networks)        ├── iNaturalist (CC-licensed)
├── Boids (flocking patterns)         ├── Wikimedia Commons (SEM, macro)
├── CCA (decay/dissolution)           └── Satellite rasters (ORNL DAAC)
└── Export PNGs                           │
         │                                ▼
         ▼                     prepare_dataset.py (stage captions)
  raster_to_controlnet.py             │
  (satellite → depth maps)            ▼
         │                     LoRA Training (ai-toolkit)
         ▼                     ├── biomass_growth LoRA
  ComfyUI (3080 LAN / Cloud)  └── biomass_decay LoRA
  ├── FLUX Depth ControlNet            │
  ├── Custom LoRA (lifecycle)          │
  ├── IPAdapter (style ref)   ◄────────┘
  └── txt2img prompt
         │
         ▼
  Output: sim skeleton → photorealistic biomass
```

## Quick Start

```bash
cd scripts
pip install -r requirements.txt

# 1. Scrape biological texture training data
python scrape_textures.py --preset lifecycle_growth --output ../datasets/raw/biomass_growth/ --limit 30
python scrape_textures.py --preset lifecycle_decay --output ../datasets/raw/biomass_decay/ --limit 30
python scrape_textures.py --preset sem_textures --output ../datasets/raw/sem/ --limit 20

# 2. Prepare LoRA dataset with lifecycle stage captions
python prepare_dataset.py -i ../datasets/raw/biomass_growth/ -o ../datasets/biomass_growth/ -t biomaesthetic --type biomass_growth
python prepare_dataset.py -i ../datasets/raw/biomass_decay/ -o ../datasets/biomass_decay/ -t biomaesthetic --type biomass_decay
# Or auto-interpolate stages across a sim frame sequence:
python prepare_dataset.py -i ../datasets/raw/physarum_seq/ -o ../datasets/lifecycle/ -t biomaesthetic --stage-range 1,5

# 3. Convert satellite/raster data to ControlNet inputs
python raster_to_controlnet.py -i vegetation_change.tif -o controlnet_input.png --mode grayscale --invert --blur 3

# 4. Process sim frames through FLUX + ControlNet + LoRA
python batch_process.py -i ../datasets/raw/dla/ -w ../workflows/flux_controlnet_depth_lora.json --host http://192.168.x.x:8188

# 5. Parameter sweep (txt2img)
python sweep_txt2img.py --host http://192.168.x.x:8188 --sweep sampler

# Legacy: export DLA frames from ZIP
python export_frames.py unzip --input path/to/ew-export.zip --output ../datasets/raw/dla/
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
│   ├── prepare_dataset.py              # LoRA dataset prep + lifecycle stage captions
│   ├── scrape_textures.py              # CC-licensed bio texture scraper (iNaturalist, Wikimedia)
│   ├── raster_to_controlnet.py         # Satellite/GeoTIFF → ControlNet PNG
│   ├── export_frames.py                # DLA frame extraction
│   ├── sweep_denoise.py                # Parameter sweep comparisons
│   ├── sweep_txt2img.py                # KSampler settings sweep
│   ├── train_config_sdxl.yaml          # ai-toolkit SDXL config
│   └── train_config_flux.yaml          # ai-toolkit Flux config
├── datasets/                 # Training datasets (img + txt pairs)
├── loras/                    # Trained LoRA checkpoints
├── outputs/                  # Generated outputs
└── docs/                     # Pipeline documentation, comparisons
```

## Hardware

| Task | 3080 (10GB) | Cloud A100 | Mac |
|------|:-----------:|:----------:|:---:|
| SDXL img2img / ControlNet | 1024px | — | — |
| SDXL + ControlNet + IPAdapter | 768px | — | — |
| Flux dev fp8 | 512px | 1024px | — |
| SDXL LoRA training (rank 16) | Yes | Faster | — |
| Flux LoRA training | — | Yes | — |
| Dataset prep / scripting | — | — | Yes |

## Models Required

**On Windows 3080:**
- `sd_xl_base_1.0.safetensors` + `sdxl_vae.safetensors`
- `control-lora-canny-rank256.safetensors` (SDXL ControlNet)
- `ip-adapter-plus_sdxl_vit-h.safetensors` + `CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors`
- `flux1-dev-fp8.safetensors` + `t5xxl_fp8_e4m3fn.safetensors` + `clip_l.safetensors` + `ae.safetensors`

**Custom nodes (install via ComfyUI Manager):**
ComfyUI-Manager, comfyui-controlnet-aux, ComfyUI_IPAdapter_plus, ComfyUI-GGUF, rgthree-comfy, was-node-suite, ComfyUI-Impact-Pack, ComfyUI-AnimateDiff-Evolved
