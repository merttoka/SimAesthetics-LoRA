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
- Preprocessor: `CannyEdgePreprocessor` (low=100, high=200, resolution=1024)
- Applied via `Apply ControlNet` node, strength 0.7

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

### Other approaches to explore
- **Inpainting**: use sim frame luminance as mask → only regenerate bright regions
- **Post-composite**: generate fully, then mask-blend back onto black using original luminance
- **Depth ControlNet**: better for organic volumetric content than Canny (edges vs spatial relationships)

---

## 4. Interesting Prompt Directions

### Sim frame + non-sim prompts (ControlNet structure transfer)
- `underwater coral reef, photorealistic, macro photography` → sim veins become coral
- `aerial city at night, glowing streets, cyberpunk` → sim network becomes streets
- `photorealistic SEM image, fungal mycelium, dense biological tissue` → Gemini's biomass proposal

### Real photo + LoRA (style transfer)
- Any photo + `simaesthetic` trigger + LoRA → rendered in sim aesthetic
- Dog photo gets organic/bioluminescent background treatment

---

## 5. Pipeline Architecture

```
                                    ┌─ LoRA (learned aesthetic)
                                    │
Sim Frame ──→ VAE Encode ──→ KSampler ──→ VAE Decode ──→ Output
    │                          ↑
    └──→ Canny/Depth ──→ ControlNet (structure preservation)

    Denoise 0.5-0.7: reimagine with structure
    LoRA strength 0.85-0.95: aesthetic intensity
    ControlNet strength 0.5-0.9: structural fidelity
```

---

## 6. Interview Talking Points Developed

### What I built
- End-to-end pipeline: Unity GPU sim → dataset prep → LoRA training → ComfyUI inference
- Automated tooling: frame extraction, smart cropping, lifecycle captioning, texture scraping
- FLUX workflow (untested due to VRAM limits) + SDXL workflow (working)

### What I learned
- LoRA trigger word binding requires aggressive caption dropout
- ControlNet is redundant when source images ARE the edge structure
- img2img with controlled denoise is more useful than txt2img+ControlNet for sim frames
- The model as selective texture synthesizer (respecting sim spatial logic) is the compelling framing

### Advanced concepts to discuss
- **Latent decay**: spatially varying the ODE solver halt point using sim dead-cell masks
- **Multi-LoRA routing**: sim output masks route between growth/decay LoRAs spatially
- **Vector field flow matching**: injecting sim velocity fields into FLUX initial noise distribution
- **Satellite data as ControlNet**: real ecological change-detection rasters as structural input

---

## 7. Unresolved / Next Steps

- [ ] FLUX LoRA training (needs A100 80GB)
- [ ] img2img + ControlNet combo (denoise as reimagine knob)
- [ ] Inpainting approach (mask from sim luminance)
- [ ] Depth ControlNet vs Canny for organic content
- [ ] Higher caption_dropout_rate training run
- [ ] Batch process comparison grid for interview artifact
- [ ] Side-by-side video: raw sim | AI-rendered
