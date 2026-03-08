# SimAesthetics Explorations Log

## Context
BFL Creative Technologist interview prep. Building ALife sim → generative AI pipeline.
Unity Edge of Chaos (Physarum/Boids) → SDXL/FLUX + LoRA + ControlNet → photorealistic biological matter.

---

## 1. Dataset Preparation

### Source material
- 91 frames from Unity Edge of Chaos recording (14336x1920, Physarum+Boids hybrid)
- Every 500th frame from 45k frame sequence

### Cropping strategy
- `--crop sample` mode: random 1024x1024 crops from middle 30% of horizontal space (`h_focus=0.3`)
- Brightness filter (`min_brightness=8.0`) rejects mostly-black crops
- 3 crops per frame → **270 training pairs**
- Captions: `simaesthetic, physarum polycephalum slime mold network, organic branching transport network, bioluminescent veins`
- **Crop coordinates now saved in manifest.json** (x, y, crop_size, source dimensions) for overlay compositing

### Key learning
- Center crop on ultrawide = always same region. Random sampling with focus region = diverse views
- Brightness filtering critical — early frames and edge regions are mostly void
- 270 images is large for LoRA (50-100 typical). Works but need to watch for overfitting

---

## 2. LoRA Training (SDXL)

### Config
- Base: `sd_xl_base_1.0`
- Rank 16, alpha 16
- 2500 steps (~9 epochs over 270 images)
- Optimizer: adamw8bit, lr: 1e-4, cosine schedule
- Trigger word: `simaesthetic`
- `caption_dropout_rate: 0.05`

### Infrastructure
- 3080 (10GB): too slow — ~80hrs estimated. Sample generation during training OOMs at 1024px
- **RunPod L40S (48GB)**: ~55 min for 2500 steps with sampling every 250 steps
- FLUX training failed on L40S — model loads fp32 (~44GB) before quantizing, OOMs. Needs A100 80GB

### Results
- LoRA clearly learned the aesthetic by step 1250 (cyan veins, amber fill, particle trails, dark bg)
- Dog test: clean dog at step 2500 — **no overfitting**
- Steps 1250-2500 are similar — LoRA converged early
- **Trigger word is weak**: `simaesthetic` alone doesn't strongly activate the style. Full caption works. Likely need higher `caption_dropout_rate` (0.1-0.15) to bind trigger word more tightly
- LoRA strength 0.85-0.95 gives visible style transfer on dogs/other subjects

### What I'd change next time
- `caption_dropout_rate: 0.15` — force stronger trigger word association
- Consider trigger-word-only captions for 50% of dataset
- Fewer images (90) with more epochs might give sharper style binding
- Try rank 32 for more expressive capacity

---

## 3. ControlNet Experiments

### Setup
- Model: `control-lora-canny-rank256.safetensors`
- Preprocessor: `CannyEdgePreprocessor` (low=237, high=255, resolution=1024)
- Applied via `ControlNetApplyAdvanced` node, strength 0.8

### Observation: Canny + sim frames is redundant
Sim frames are already edges/lines on black. Canny edge extraction → regeneration produces near-identical output. Round-tripping structure through ControlNet doesn't add value when the source IS edges.

### Observation: ControlNet fills void regions
ControlNet txt2img generates content everywhere — black void regions get filled with prompt-derived content. Not desirable for sim aesthetic where void = entropy/emptiness.

### Solution: img2img mode instead of txt2img
Replace `Empty Latent Image` with `VAE Encode` of the sim frame:
```
Load Image → VAE Encode → KSampler (denoise 0.5-0.7)
```
- Starts from actual image, not noise
- Black stays black, detail added only to existing structure
- Denoise slider = "how much to reimagine" knob
- 0.3 = barely changes. 0.7 = rich texture, void stays dark

### Key insight
This reframes the pipeline: the diffusion model is a **selective texture synthesizer** respecting the sim's spatial logic. Dark = void = entropy. Bright = structure = life. The model only "grows" where the algorithm says there's life.

---

## 4. Parameter Insights (Tested)

### Best SDXL settings for sim frames
| Parameter | Value | Notes |
|-----------|-------|-------|
| LoRA strength | 0.85-0.95 | Below 0.8 = barely visible. Above 1.0 = artifacts |
| cfg | 6.0 | 7.0 works but 6.0 gives more natural results |
| denoise | 0.5-0.7 | **The key knob.** 0.5 = subtle enhancement. 0.7 = heavy reimagining. Both preserve void |
| Canny low/high | 237/255 | High thresholds because sim frames are bright-on-black — low thresholds pick up noise |
| ControlNet strength | 0.7-0.8 | Higher = more structural fidelity, less creative freedom |
| sampler | dpmpp_2m_sde | karras scheduler |
| steps | 30 | Diminishing returns past 30 |

### img2img vs ControlNet for sim frames
- **img2img alone** (denoise 0.6): best void preservation, good texture. **Use this as default**
- **img2img + ControlNet**: tighter structure lock. Use when denoise is high and output drifts from input layout
- **ControlNet alone (txt2img)**: fills void regions. Not ideal for sim aesthetic
- **ControlNet Canny on sim frames**: redundant — sim IS edges. Depth ControlNet may be better (untested)

### Trigger word behavior
- `simaesthetic` alone: weak activation. Needs full descriptive prompt alongside
- `simaesthetic` + full caption + LoRA 0.9: strong activation
- Higher `caption_dropout_rate` (0.10-0.15) would fix this in retraining

---

## 5. Workflow Architecture

### Available workflows (API + UI format)
| Workflow | Purpose | Status |
|----------|---------|--------|
| `sdxl_img2img_lora` | img2img + LoRA — **primary workflow** | Tested, working |
| `sdxl_controlnet_lora` | img2img + Canny ControlNet + LoRA | Tested, working |
| `sdxl_img2img` | img2img baseline (no LoRA) | Working |
| `sdxl_controlnet_canny` | ControlNet only (no LoRA) | Working |
| `sdxl_controlnet_ipa` | ControlNet + IPAdapter (style ref chaining) | Untested |
| `flux_controlnet_depth_lora` | FLUX + Depth CN + LoRA | Untested (needs A100) |

### Pipeline flow
```
                                    ┌─ LoRA (learned aesthetic)
                                    │
Sim Frame ──→ VAE Encode ──→ KSampler ──→ VAE Decode ──→ Output
    │                          ↑
    └──→ Canny/Depth ──→ ControlNet (optional structure lock)

batch_process.py → runs N frames through ComfyUI API
overlay_composite.py → pastes AI crops back onto ultrawide frames
make_grid.py → side-by-side comparison grids
```

---

## 6. Batch Processing Infrastructure

### Scripts
- `batch_process.py`: sends frames to ComfyUI over LAN, saves outputs locally. Supports `--limit`, `--denoise`, `--cfg`, `--seed`, parallel/chained modes
- `make_grid.py`: side-by-side grids from pairs or directories
- `overlay_composite.py`: composites AI crops back onto ultrawide source frames using manifest coordinates. Supports `--side-by-side` and `--opacity`
- `pod.sh`: RunPod SSH/SCP helper for training

### Overlay video pipeline
1. `prepare_dataset.py` saves crop coordinates in manifest.json
2. `batch_process.py` generates AI outputs
3. `overlay_composite.py` pastes AI crops back at original coordinates on ultrawide frames
4. `ffmpeg` encodes to video

---

## 7. Interview Talking Points

### What I built
- End-to-end pipeline: Unity GPU sim → dataset prep → LoRA training → ComfyUI inference → overlay compositing
- Automated tooling: smart cropping with coordinate tracking, batch processing, comparison grids, overlay video
- SDXL LoRA trained on RunPod L40S (~55 min)

### What I learned
- LoRA trigger word binding requires aggressive caption dropout (0.05 too low, need 0.15)
- ControlNet is redundant when source images ARE the edge structure
- img2img with controlled denoise is the core technique — model as selective texture synthesizer
- Dark = void = entropy. The model only "grows" where the algorithm says there's life

### Advanced concepts to discuss
- **Latent decay**: spatially varying the ODE solver halt point using sim dead-cell masks
- **Multi-LoRA routing**: sim output masks route between growth/decay LoRAs spatially
- **Vector field flow matching**: injecting sim velocity fields into FLUX initial noise distribution
- **Satellite data as ControlNet**: real ecological change-detection rasters as structural input

---

## 8. FLUX LoRA Training — Should We?

### Pros
- FLUX produces significantly higher quality/coherence than SDXL
- Flow matching (FLUX) vs DDPM (SDXL) — better for fine detail
- T5 text encoder understands complex prompts better — trigger word may bind more naturally
- BFL interview — demonstrating FLUX mastery is directly relevant

### Cons
- Needs A100 80GB (~$2-3/hr on RunPod). FLUX loads fp32 before quantizing
- No trained FLUX LoRA yet — would need to re-prep dataset (or reuse same 270 images)
- ControlNet ecosystem for FLUX is less mature than SDXL
- Current SDXL results already look good

### Recommendation
**Yes, train FLUX** if time permits. Use same 270-image dataset. Key changes:
- `caption_dropout_rate: 0.15` (fix trigger word binding)
- `noise_scheduler: flowmatch` (not ddpm)
- `guidance_scale: 3.5` (FLUX uses lower cfg)
- RunPod A100 80GB, estimate ~1-2 hrs
- Compare FLUX vs SDXL outputs in grid — strong interview artifact

### If no time for FLUX training
Focus on maximizing SDXL output quality:
- Retrain SDXL with `caption_dropout_rate: 0.15` for stronger trigger
- Generate more diverse outputs (different prompts, denoise values)
- Build rich comparison grid + overlay video

---

## 9. Unresolved / Next Steps

- [ ] FLUX LoRA training (A100 80GB, ~$2-3/hr)
- [ ] Retrain SDXL with `caption_dropout_rate: 0.15`
- [x] ~~Batch process comparison grid~~ → batch_process.py + make_grid.py working
- [x] ~~Crop coordinate tracking~~ → manifest.json now saves x/y/crop_size
- [ ] Overlay composite video (overlay_composite.py ready, needs re-prepped dataset with coords)
- [ ] Depth ControlNet vs Canny for organic content
- [ ] IPAdapter chained mode (style continuity across frames)
- [ ] Non-sim prompts: coral reef, aerial city, mycelium over sim structure
- [ ] Try inpainting approach (luminance mask → only regenerate bright regions)
