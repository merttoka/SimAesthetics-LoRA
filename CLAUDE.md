# COMFY_SimAesthetics

ComfyUI + LoRA pipeline for generative simulation aesthetics. Replaces BFL API with local/LAN ComfyUI.

## Architecture
- **Mac** runs scripts (Python), sends to ComfyUI on **Windows 3080** via LAN
- ComfyUI API at `http://<windows-ip>:8188`
- Cloud (RunPod L40S) for SDXL LoRA training

## Key Scripts
- `comfyui_client.py` — API client (replaces bfl.ts)
- `batch_process.py` — parallel/chained frame processing
- `prepare_dataset.py` — LoRA dataset prep (random crops, brightness filter, captions)
- `make_grid.py` — side-by-side comparison grid builder
- `pod.sh` — RunPod SSH/SCP helper
- `overlay_composite.py` — composite AI crops back onto ultrawide frames (uses manifest coords)
- `sweep_denoise.py` — parameter sweep with comparison grids (1D/2D, ComfyUI)
- `sweep_txt2img.py` — KSampler settings sweep (ComfyUI)
- `flux_sample.py` — FLUX LoRA inference: txt2img, img2img, batch directory
- `flux_sweep.py` — FLUX parameter sweeps (denoise, LoRA strength, steps)

## Workflows (ComfyUI API format JSON)
- `sdxl_img2img.json` — basic img2img (no LoRA, baseline)
- `sdxl_img2img_lora.json` — **img2img + LoRA (best approach)** — preserves void, adds texture
- `sdxl_controlnet_canny.json` — ControlNet Canny (no LoRA)
- `sdxl_controlnet_lora.json` — img2img + ControlNet Canny + LoRA
- `sdxl_controlnet_ipa.json` — ControlNet + IPAdapter (style ref from prev frame)
- `flux_controlnet_depth_lora.json` — FLUX + Depth ControlNet + LoRA (untested, needs A100)

## LoRA Training
- Trigger word: `simaesthetic`
- Dataset: 270 crops from 91 ultrawide Physarum/Boids sim frames
- Trained: SDXL rank 16, 2500 steps, RunPod L40S ~55min
- Best settings: LoRA 0.85-0.95, cfg 6.0, denoise 0.5-0.7

## Node ID conventions in workflows
- `1` = CheckpointLoader, `3` = KSampler, `6` = positive prompt, `10` = input image, `12` = style ref, `40` = LoRA

## Training configs
- `train_config_sdxl.yaml` — ai-toolkit config, local 3080
- `train_config_sdxl_runpod.yaml` — ai-toolkit config, RunPod L40S
- `train_config_flux.yaml` — ai-toolkit config, cloud A100 (untested)

## Dependencies
- Python 3.11+, Pillow, websocket-client
- Optional: transformers+torch (captioning), playwright (frame capture)
