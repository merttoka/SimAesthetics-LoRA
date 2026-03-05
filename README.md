# COMFY_SimAesthetics

ComfyUI + LoRA generative media pipeline for simulation aesthetics. Rebuilds the [BFL_FLUXdemos/emergent-worlds](../BFL_FLUXdemos) pipeline locally with ControlNet conditioning, IPAdapter style transfer, and custom LoRAs.

## Pipeline

```
DLA Simulation → Frame Export → ComfyUI (3080 LAN / RunPod cloud)
                                  ├── ControlNet (structure preservation)
                                  ├── IPAdapter (temporal style coherence)
                                  ├── Custom LoRA (trained aesthetic)
                                  └── txt2img prompt (from prompts.ts)
                                → Output images / video
```

## Quick Start

```bash
cd scripts
pip install -r requirements.txt

# Export frames from existing ZIP
python export_frames.py unzip --input path/to/ew-export.zip --output ../datasets/raw/dla/

# Process through ComfyUI
python batch_process.py -i ../datasets/raw/dla/ -w ../workflows/sdxl_img2img.json --host http://192.168.x.x:8188

# Parameter sweep
python sweep_denoise.py -i frame.png -w ../workflows/sdxl_img2img.json -p 3.denoise -v 0.3,0.5,0.7,0.9 --grid

# Prepare LoRA dataset
python prepare_dataset.py -i ../datasets/raw/dla/ -o ../datasets/dla_aesthetic/ -t dlaaesthetic --size 1024
```

## Structure

```
COMFY_SimAesthetics/
├── workflows/          # ComfyUI workflow JSONs (API format)
│   ├── sdxl_img2img.json
│   ├── sdxl_controlnet_canny.json
│   ├── sdxl_controlnet_ipa.json    # Full BFL API equivalent
│   └── sdxl_controlnet_lora.json
├── scripts/            # Automation & tooling (runs on Mac)
│   ├── comfyui_client.py           # ComfyUI API client
│   ├── batch_process.py            # Batch frame processing
│   ├── prepare_dataset.py          # LoRA dataset prep + captioning
│   ├── export_frames.py            # DLA frame extraction
│   ├── sweep_denoise.py            # Parameter sweep comparisons
│   ├── train_config_sdxl.yaml      # ai-toolkit SDXL config
│   └── train_config_flux.yaml      # ai-toolkit Flux config
├── datasets/           # Training datasets (img + txt pairs)
├── loras/              # Trained LoRA checkpoints
├── outputs/            # Generated outputs
└── docs/               # Pipeline documentation, comparisons
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
