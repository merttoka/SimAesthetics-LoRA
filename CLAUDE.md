# COMFY_SimAesthetics

ComfyUI + LoRA pipeline for generative simulation aesthetics. Replaces BFL API with local/LAN ComfyUI.

## Architecture
- **Mac** runs scripts (Python), sends to ComfyUI on **Windows 3080** via LAN
- ComfyUI API at `http://<windows-ip>:8188`
- Cloud (RunPod A100) for Flux training + high-res inference

## Key Scripts
- `comfyui_client.py` — API client (replaces bfl.ts)
- `batch_process.py` — parallel/chained frame processing
- `prepare_dataset.py` — LoRA dataset prep, lifecycle stage captions, Florence-2 auto-captioning
- `scrape_textures.py` — CC-licensed bio texture scraper (iNaturalist, Wikimedia Commons)
- `raster_to_controlnet.py` — satellite/GeoTIFF → ControlNet-ready PNG (grayscale/viridis/binary)
- `export_frames.py` — DLA frame extraction (ZIP or Playwright)
- `sweep_denoise.py` — parameter sweep with comparison grids
- `sweep_txt2img.py` — KSampler settings sweep (sampler, cfg, steps, scheduler)

## Workflows (ComfyUI API format JSON)
- `sdxl_img2img.json` — basic img2img
- `sdxl_controlnet_canny.json` — ControlNet canny
- `sdxl_controlnet_ipa.json` — ControlNet + IPAdapter (full BFL API equivalent)
- `sdxl_controlnet_lora.json` — ControlNet + custom LoRA
- `flux_controlnet_depth_lora.json` — FLUX + Depth ControlNet + LoRA (lifecycle prototype)

## Biomass LoRA Training
- Trigger word: `biomaesthetic`
- 5 lifecycle stages: pristine growth → mature → aging → decay → necrotic
- `--stage-range 1,5` in prepare_dataset.py auto-interpolates captions across frame sequences
- Dataset types: `biomass_growth`, `biomass_mature`, `biomass_aging`, `biomass_decay`, `biomass_necrotic`, `sem`, `fungal`, `rust`

## Node ID conventions in workflows
- `1` = CheckpointLoader, `3` = KSampler, `6` = positive prompt, `10` = input image, `12` = style ref

## Training
- `train_config_sdxl.yaml` — ai-toolkit config, local 3080
- `train_config_flux.yaml` — ai-toolkit config, cloud A100

## Dependencies
- Python 3.11+, Pillow, websocket-client
- Optional: transformers+torch (captioning), playwright (frame capture)
