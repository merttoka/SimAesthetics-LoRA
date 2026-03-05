# COMFY_SimAesthetics

ComfyUI + LoRA pipeline for generative simulation aesthetics. Replaces BFL API with local/LAN ComfyUI.

## Architecture
- **Mac** runs scripts (Python), sends to ComfyUI on **Windows 3080** via LAN
- ComfyUI API at `http://<windows-ip>:8188`
- Cloud (RunPod A100) for Flux training + high-res inference

## Key Scripts
- `comfyui_client.py` — API client (replaces bfl.ts)
- `batch_process.py` — parallel/chained frame processing
- `prepare_dataset.py` — LoRA training dataset prep
- `export_frames.py` — DLA frame extraction (ZIP or Playwright)
- `sweep_denoise.py` — parameter sweep with comparison grids

## Workflows (ComfyUI API format JSON)
- `sdxl_img2img.json` — basic img2img
- `sdxl_controlnet_canny.json` — ControlNet canny
- `sdxl_controlnet_ipa.json` — ControlNet + IPAdapter (full BFL API equivalent)
- `sdxl_controlnet_lora.json` — ControlNet + custom LoRA

## Node ID conventions in workflows
- `1` = CheckpointLoader, `3` = KSampler, `6` = positive prompt, `10` = input image, `12` = style ref

## Training
- `train_config_sdxl.yaml` — ai-toolkit config, local 3080
- `train_config_flux.yaml` — ai-toolkit config, cloud A100

## Dependencies
- Python 3.11+, Pillow, websocket-client
- Optional: transformers+torch (captioning), playwright (frame capture)
