# SimAesthetics Architecture

## Pipeline

```
Unity Edge of Chaos (Physarum + Boids)
└── 45k frames → every 500th → 91 ultrawide (14336x1920)
         │
    prepare_dataset.py
    ├── random 1024² crops from middle 30% (h_focus=0.3)
    ├── brightness filter (min_brightness=8.0)
    ├── crop coordinates saved in manifest.json
    └── 3 crops/frame → 270 training pairs
         │
         ▼
    ai-toolkit LoRA (SDXL, rank 16, 2500 steps)
    ├── trigger: "simaesthetic"
    ├── RunPod L40S ~55min
    └── 6 checkpoints (every 250 steps)
         │
         ▼
    ComfyUI (Windows 3080 LAN, http://<comfyui-host>:8188)
    ├── batch_process.py → N frames through workflow
    ├── sweep_denoise.py → parameter sweeps with grids
    └── make_grid.py → comparison grids
         │
         ▼
    overlay_composite.py → AI crops back onto ultrawide frames → video
```

## Scripts

| Script | Purpose |
|--------|---------|
| `prepare_dataset.py` | Crop + caption sim frames. Saves coordinates in manifest.json |
| `batch_process.py` | Send frames to ComfyUI API, save outputs. `--limit`, `--denoise`, `--lora-strength`, `--mode` |
| `sweep_denoise.py` | Parameter sweep: 1D or 2D (matrix), fixed seed, labeled grid. `-p`/`-p2` |
| `make_grid.py` | Comparison grids. `--timelapse`, `--count`/`--iter`, auto-wrap layout |
| `overlay_composite.py` | Paste AI crops back onto ultrawide frames at original coords |
| `comfyui_client.py` | ComfyUI HTTP/WebSocket API client |
| `pod.sh` | RunPod SSH/SCP helper (connect, upload, download, loras, samples) |
| `make_grid.py` | Comparison grids with timelapse, batching, auto-wrap |

## Workflows

All in ComfyUI API format (flat node dicts). No `_comment` keys (Impact Pack crashes on them).

| Workflow | Description | Key params |
|----------|-------------|-----------|
| `sdxl_img2img_lora.json` | **Primary.** img2img + LoRA | denoise 0.6, cfg 6, LoRA 0.9 |
| `sdxl_controlnet_lora.json` | img2img + Canny ControlNet + LoRA | + CN strength 0.8, Canny 237/255 |
| `sdxl_img2img.json` | img2img baseline (no LoRA) | denoise 0.6, cfg 6 |
| `sdxl_controlnet_canny.json` | ControlNet only (no LoRA) | CN strength 0.8, Canny 237/255 |
| `sdxl_controlnet_ipa.json` | ControlNet + IPAdapter (style chaining) | untested |
| `flux_controlnet_depth_lora.json` | FLUX + Depth CN + LoRA | untested, needs A100 |

UI-format versions (for drag-and-drop into ComfyUI canvas):
- `ui_sdxl_img2img_lora.json`
- `ui_sdxl_controlnet_lora.json`

## Node ID Conventions

| ID | Node | Sweepable fields |
|----|------|-----------------|
| `1` | CheckpointLoaderSimple | `ckpt_name` |
| `2` | CLIPTextEncode (negative) | `text` |
| `3` | KSampler | `seed`, `steps`, `cfg`, `sampler_name`, `scheduler`, `denoise` |
| `5` | VAEEncode | — |
| `6` | CLIPTextEncode (positive) | `text` |
| `9` | SaveImage | `filename_prefix` |
| `10` | LoadImage | `image` |
| `15` | CannyEdgePreprocessor | `low_threshold`, `high_threshold`, `resolution` |
| `16` | ControlNetLoader | `control_net_name` |
| `20` | ControlNetApplyAdvanced | `strength`, `start_percent`, `end_percent` |
| `40` | LoraLoader | `lora_name`, `strength_model`, `strength_clip` |

## Sweepable Parameters

### sdxl_img2img_lora.json

| Parameter | Node.Field | Default | Sweep range |
|-----------|-----------|---------|-------------|
| Denoise | `3.denoise` | 0.6 | 0.4–0.95 |
| CFG | `3.cfg` | 6.0 | 3–12 |
| Steps | `3.steps` | 30 | 15–50 |
| LoRA strength (model) | `40.strength_model` | 0.9 | 0.5–1.2 |
| LoRA strength (clip) | `40.strength_clip` | 0.9 | 0.5–1.2 |

### sdxl_controlnet_lora.json (all above, plus)

| Parameter | Node.Field | Default | Sweep range |
|-----------|-----------|---------|-------------|
| ControlNet strength | `20.strength` | 0.8 | 0.3–1.0 |
| Canny low threshold | `15.low_threshold` | 237 | 50–237 |
| Canny high threshold | `15.high_threshold` | 255 | 200–255 |

### Sweep commands

```bash
# Denoise sweep (most important — denser images need higher denoise)
python scripts/sweep_denoise.py \
  -i datasets/sim_aesthetic_2/img_010.png \
  -w workflows/sdxl_img2img_lora.json \
  --host http://<comfyui-host>:8188 \
  -p 3.denoise --range 0.5,0.95,6 \
  -o outputs/sweep_denoise/

# LoRA strength
python scripts/sweep_denoise.py ... -p 40.strength_model --range 0.5,1.2,5

# CFG
python scripts/sweep_denoise.py ... -p 3.cfg --range 3,12,4

# ControlNet strength
python scripts/sweep_denoise.py \
  -w workflows/sdxl_controlnet_lora.json \
  -p 20.strength --range 0.3,1.0,5

# 2D sweep: LoRA strength × ControlNet strength (matrix grid)
python scripts/sweep_denoise.py \
  -i datasets/sim_aesthetic_2/img_010.png \
  -w workflows/sdxl_controlnet_lora.json \
  --host http://<comfyui-host>:8188 \
  -p 40.strength_model --range 0.15,0.5,4 \
  -p2 20.strength --range2 0.3,0.8,4 \
  -o outputs/sweep_lora_x_cn/
```

## Grid Creation

```bash
# Basic: first 8 pairs
python scripts/make_grid.py -l datasets/ -r outputs/ --end 8 -o grid.png

# Timelapse: one crop per source frame, sorted by frame number
python scripts/make_grid.py \
  -l datasets/sim_aesthetic_2/ -r outputs/controlnet-lora/ \
  --timelapse -m datasets/sim_aesthetic_2/manifest.json \
  -o grid.png

# Batched: 3 grids of 6 frames each
python scripts/make_grid.py ... --count 6 --iter 3 -o grid.png

# Smaller cells, more columns
python scripts/make_grid.py ... --size 256 --max-cols 12 -o grid.png
```

Auto-wraps into rows when pairs > `--max-cols` (default 8), keeping cols >= rows for landscape orientation.

## Key Findings

- **img2img > txt2img**: VAE Encode + denoise 0.5-0.7 preserves void. txt2img fills everything
- **Canny ControlNet redundant on sim frames**: sim IS edges. Round-tripping adds nothing
- **Denoise is content-dependent**: sparse frames work at 0.6. Dense frames need 0.8-0.9
- **Trigger word weak at caption_dropout 0.05**: need 0.10-0.15 for strong binding
- **No `_comment` in API JSONs**: Impact Pack's onprompt hook crashes on non-node keys
- **No `Reroute` nodes in API JSONs**: same Impact Pack crash

## Training Configs

| Config | Target | Hardware | Time | Caption dropout | Status |
|--------|--------|----------|------|-----------------|--------|
| `train_config_sdxl.yaml` | SDXL rank 16 | Local 3080 | ~80hr | 0.05 | v1 done (impractical) |
| `train_config_sdxl_runpod.yaml` | SDXL v2 rank 16 | RunPod A40 48GB | ~55min | **0.15** | **Training** |
| `train_config_flux.yaml` | FLUX rank 16 | RunPod A100 80GB | ~1-2hr | **0.15** | **Training** |

### Training round 2 pods
```bash
# Pod management (pod.sh supports both)
./scripts/pod.sh flux connect     # A100 80GB (IP in pod_config.sh)
./scripts/pod.sh sdxl connect     # A40 48GB  (IP in pod_config.sh)
./scripts/pod.sh flux loras       # download FLUX checkpoints
./scripts/pod.sh sdxl loras       # download SDXL v2 checkpoints
./scripts/pod.sh flux stop
./scripts/pod.sh sdxl stop
```

### GPU selection for LoRA training
- Multi-GPU doesn't help — ai-toolkit uses `cuda:0` only
- FLUX needs 80GB on one card (loads fp32 before quantizing)
- A100 SXM 80GB: $1.22/hr spot (cheapest for FLUX)
- 2x A40: $0.40/hr spot (cheapest for SDXL, 48GB/card)
- Volume disk: 50GB minimum

## What to try next

- [ ] Download + compare FLUX vs SDXL v2 outputs in grid
- [ ] Denoise sweep on sim_aesthetic_2 (dense frames need higher denoise)
- [ ] Depth ControlNet instead of Canny for volumetric organic content
- [ ] IPAdapter chained mode for style continuity across frame sequences
- [ ] Non-sim prompts over sim structure: coral reef, aerial city, mycelium
- [ ] Overlay video: re-run prepare_dataset for coords → batch process → composite → ffmpeg
