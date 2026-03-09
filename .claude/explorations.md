# SimAesthetics Explorations Log

## Context
ALife sim → generative AI pipeline. Exploring LoRA fine-tuning and img2img workflows for simulation aesthetics.
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
| LoRA strength | **0.3-0.4** (creative) or 0.85-0.95 (faithful) | See "LoRA as creativity dial" below |
| cfg | 6.0 | 7.0 works but 6.0 gives more natural results |
| denoise | 0.5-0.7 (sparse) or 0.76 (dense) | Content-dependent — see below |
| Canny low/high | 237/255 | High thresholds because sim frames are bright-on-black — low thresholds pick up noise |
| ControlNet strength | 0.7-0.8 | Higher = more structural fidelity, less creative freedom |
| sampler | dpmpp_2m_sde | karras scheduler |
| steps | 30 | Diminishing returns past 30 |

### LoRA strength as a creativity dial
- **Big insight**: LoRA training doesn't produce a fixed output — it creates a **continuous aesthetic axis**
- **Low strength (0.3-0.4)**: SDXL reimagines freely, LoRA provides subtle guidance. More creative, more surprising results. Organic textures that don't look like the training data
- **High strength (0.85-0.95)**: output closely matches training data aesthetic. Faithful physarum reproduction
- The v1 batch (weak trigger binding at caption_dropout 0.05) accidentally produced a low-strength-like effect — outputs looked creative and surprising
- v2 (caption_dropout 0.15) + strength 0.35 reproduces that same creative look intentionally
- **LoRA strength is the primary creative knob** — more important than denoise for controlling output character

### img2img vs ControlNet for sim frames
- **img2img alone** (denoise 0.6): best void preservation, good texture. **Use this as default**
- **img2img + ControlNet**: tighter structure lock. Use when denoise is high and output drifts from input layout
- **ControlNet alone (txt2img)**: fills void regions. Not ideal for sim aesthetic
- **ControlNet Canny on sim frames**: redundant — sim IS edges. Depth ControlNet may be better (untested)

### Denoise is content-dependent
- **Sparse frames** (sim_aesthetic, lots of void): denoise 0.5-0.7 works well. Strong transformation
- **Dense frames** (sim_aesthetic_2, filled structure): denoise 0.6 produces near-identical output. **0.76 is the sweet spot** — enough transformation, preserves structure
- 0.9+ was too aggressive for full batch despite looking interesting in single-image sweeps
- img2img starts from the encoded image — denser input = less room for the model to reimagine at low denoise
- **Always sweep denoise for new source material**

### Trigger word behavior (v2 update)
- v1 (caption_dropout 0.05): `simaesthetic` alone = weak activation. Needed full descriptive prompt
- **v2 (caption_dropout 0.15): trigger word binds strongly** — `simaesthetic` alone activates the style
- Higher caption_dropout forces the model to associate the trigger word with the visual style, not the full caption

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

## 6. Batch Processing & Sweep Infrastructure

### Batch processing
- `batch_process.py`: sends frames to ComfyUI over LAN, saves outputs locally
  - `--limit N` to process only first N frames
  - `--denoise`, `--cfg`, `--seed` override workflow defaults
  - `--mode parallel|chained` (chained uses prev output as style ref)
  - Host: `--host http://<comfyui-host>:8188` (no trailing slash!)

### Parameter sweeps
- `sweep_denoise.py`: vary one parameter, fixed seed, auto-generates labeled comparison grid
  - `--range start,end,steps` for evenly spaced values (e.g. `--range 0.5,0.95,6`)
  - `--values` for explicit values (e.g. `--values 0.5,0.7,0.9`)
  - Grid shows original + all sweep results side by side with labels
  - Sweepable params per workflow documented in `architecture.md`

### Grid creation (`make_grid.py`)
- Layout: vertical pairs (sim top, AI bottom), extending horizontally
- `--timelapse`: reads manifest.json, picks one crop per source frame, sorted by frame number. Labels show frame numbers (e.g. f0500, f1000)
- `--count N --iter M`: generate M grids of N pairs each
- `--start` / `--end`: index range selection
- Auto-wraps into rows when pairs > `--max-cols` (default 8), always cols >= rows
- Range appended to filename: `grid_0-8.png`, `grid_8-16.png`

### Overlay video pipeline
1. `prepare_dataset.py` saves crop coordinates in manifest.json (x, y, crop_size, source dims)
2. `batch_process.py` generates AI outputs
3. `overlay_composite.py` pastes AI crops back at original coordinates on ultrawide frames
4. `ffmpeg` encodes to video

### ComfyUI API gotchas discovered
- **No `_comment` keys in workflow JSON**: Impact Pack's `onprompt` hook iterates all keys expecting node dicts — non-node keys crash it with 500
- **No `Reroute` nodes**: same Impact Pack crash — wire outputs directly to inputs
- **No trailing `/` on host URL**: causes double-slash in API paths
- **ComfyUI needs restart to pick up new LoRA files** in `models/loras/`
- **Impact Pack error is silent**: server returns generic "500 Internal Server Error" with no details

---

## 7. Training Round 2 — FLUX + SDXL Retrain (In Progress)

### What changed from v1
- `caption_dropout_rate: 0.05 → 0.15` — fix weak trigger word binding
- SDXL job renamed to `sim_aesthetic_sdxl_v2`
- FLUX training attempted for first time

### Parallel training setup
Running simultaneously on separate RunPod pods:

| | SDXL v2 | FLUX |
|--|---------|------|
| Pod | 2x A40 (48GB/card, using 1) | A100 SXM 80GB |
| Cost | $0.40/hr spot | $1.22/hr spot |
| Config | `train_config_sdxl_runpod.yaml` | `train_config_flux.yaml` |
| Job name | `sim_aesthetic_sdxl_v2` | `sim_aesthetic_flux` |
| Est. time | ~55 min | ~1-2 hrs |
| Status | **Training** | **Training** |

### FLUX training notes
- Model: `black-forest-labs/FLUX.1-dev` (open weights, trainable)
- NOT FLUX Klein (API-only, no LoRA training)
- Required HF auth login (`huggingface-cli login`) — gated model
- `noise_scheduler: flowmatch` (not ddpm)
- `guidance_scale: 3.5` (FLUX uses lower cfg than SDXL's 6-7)
- `quantize: true` — fp8 to fit A100 80GB
- FLUX loads fp32 before quantizing — L40S 48GB OOMs, must use A100 80GB

### Pod management
```bash
# Updated pod.sh supports both pods:
./scripts/pod.sh flux connect    # SSH into FLUX pod
./scripts/pod.sh sdxl connect    # SSH into SDXL pod
./scripts/pod.sh flux loras      # Download FLUX checkpoints
./scripts/pod.sh sdxl loras      # Download SDXL v2 checkpoints
./scripts/pod.sh flux stop       # Stop FLUX pod
./scripts/pod.sh sdxl stop       # Stop SDXL pod
```

### GPU selection learnings
- Multi-GPU pods don't help for LoRA training — ai-toolkit uses `cuda:0` only
- 4x RTX 4090 (24GB each) can't fit FLUX on one card despite 96GB total
- A100 SXM 80GB is the cheapest single-card option for FLUX ($1.22/hr spot)
- 2x A40 at $0.40/hr spot is cheapest for SDXL (48GB/card, only need one)
- Volume disk: 50GB minimum (20GB too small for FLUX model + dataset + checkpoints)

---

## 8. SDXL v2 Results & Denoise Tuning

### SDXL v2 training complete
- caption_dropout 0.15 (up from 0.05) — trigger word binding should be stronger
- Downloaded all checkpoints (1250–2250 + final) to `loras/`
- Workflows updated to `sim_aesthetic_sdxl_v2.safetensors`

### Denoise sweep on sim_aesthetic_2
- Swept 0.5–0.95 in 6 steps on dense frames
- **0.9-0.92 looked interesting** in sweep but too aggressive for full batch — lost structure
- **0.76 is the sweet spot** for dense frames: enough transformation to add organic texture, preserves recognizable structure
- Sparse frames (sim_aesthetic) still work well at 0.6
- **Takeaway**: denoise is dataset-dependent, always sweep before batch

### FLUX training
- Running on A100 80GB, ~14.5s/step (2500 steps ≈ 10hrs)
- Had to fix: PyTorch 2.4→2.6 (enable_gqa), torchaudio ABI mismatch, disable sample gen (tokenizer crash on neg prompt)
- Checkpoints saving every 500 steps
- Cost: ~$12 at $1.22/hr spot

---

## 9. Unresolved / Next Steps

### Done
- [x] Batch processing pipeline (batch_process.py + make_grid.py)
- [x] Crop coordinate tracking in manifest.json
- [x] Parameter sweep tooling (sweep_denoise.py)
- [x] Timelapse grid mode (one crop per source frame, sorted by time)
- [x] Auto-wrapping grid layout (cols >= rows, landscape orientation)
- [x] UI-format workflows for ComfyUI canvas drag-and-drop
- [x] Fixed Impact Pack crashes (removed _comment keys, Reroute nodes)
- [x] Batch run on sim_aesthetic (270 frames, img2img_lora) — strong results
- [x] Batch run on sim_aesthetic_2 — identified denoise too low for dense content
- [x] SDXL v2 retrain — caption_dropout 0.15, A40 48GB, complete
- [x] Denoise sweep on sim_aesthetic_2 — 0.76 optimal for dense frames
- [x] Workflows updated to v2 LoRA

### Done (this session)
- [x] Batch run on sim_aesthetic_2 with v2 LoRA, denoise 0.76, lora-strength 0.35
- [x] 2D sweep: LoRA strength × ControlNet strength matrix
- [x] Timelapse grids from sim_aesthetic_2 v2 batch
- [x] Overlay composite frames (side-by-side + variations mode)
- [x] `flux_sample.py` — diffusers-based FLUX inference script (no ComfyUI needed)

### In progress
- [~] FLUX LoRA training — A100 80GB, ~10hrs, running

### To do
- [ ] Download + test FLUX checkpoints when training completes
- [ ] Generate FLUX samples on A100 pod via `flux_sample.py`
- [ ] Compare FLUX vs SDXL v2 outputs in grid

### Nice to have
- [ ] Depth ControlNet vs Canny for organic content
- [ ] IPAdapter chained mode (style continuity across frames)
- [ ] Non-sim prompts: coral reef, aerial city, mycelium over sim structure
