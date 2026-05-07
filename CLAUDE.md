# perception_models — CLAUDE.md

## Repository Overview

Meta's **Perception Models** repo. Two model families:

- **PLM** (Perception Language Model) — generative, vision+LLM, used for free-form descriptions
- **PE-Core** (Perception Encoder) — CLIP-style contrastive model, used for zero-shot classification

---

## Active Branch

```
claude/integrate-perception-models-ZWoIC
```

Always develop and push to this branch. Never push to main/master.

---

## Key File Paths

| File | Purpose |
|------|---------|
| `apps/pe/pe_description_gen.py` | PLM generative M/S/A description per identity per window |
| `apps/pe/pe_video_classify.py` | PE-Core zero-shot M/S/A classifier (new script) |
| `apps/pe/pe_attn_bias.py` | Attention bias injection (monkey-patches SDPA) |
| `apps/pe/pe_feature_similarity_viz.py` | PE-Core cosine similarity visualizer |
| `core/vision_encoder/pe.py` | `VisionTransformer`, `TextTransformer`, `CLIP` classes |
| `core/vision_encoder/config.py` | `PE_VISION_CONFIG`, `PE_TEXT_CONFIG` dicts, `PEConfig` dataclass |
| `core/vision_encoder/tokenizer.py` | `SimpleTokenizer` — CLIP BPE tokenizer |
| `core/vision_encoder/transforms.py` | `get_image_transform(image_size)` |
| `apps/plm/generate.py` | `PackedCausalTransformerGenerator`, `KVCache` |

---

## Model Architecture

### PE-Core (CLIP-style)
- `CLIP.from_config("PE-Core-G14-448", pretrained=True)` — best model
- Vision: ViT G/14, width=1536, 50 layers, image_size=448, pool_type="attn", output_dim=1280
- Text: width=1280, 24 layers, **context_length=72** (NOT 77 — PE uses 72, not OpenAI CLIP's 77)
- `pe_model.encode_image(tensor, normalize=True)` → `[B, 1280]`
- `pe_model.encode_text(tokens, normalize=True)` → `[B, 1280]`
- `pe_model.encode_video(video, normalize=True)` → `[B, 1280]` (B,N,C,H,W input)
- Tokenizer: `SimpleTokenizer(context_length=pe_model.context_length)` — always read ctx_len from model

### PE-Lang (used inside PLM)
- Derived from PE-Core-G14-448: layers=47, pool_type="none", output_dim=None, ls_init_value=0.1
- **Cannot reconstruct PE-Core features from PE-Lang** — different architecture/weights

### PLM-8B
- PE-Lang-G14-448 (47 layers) + MLPProjector + LLaMA LLM
- `pooling_ratio=1` (NOT 2) → patch grid = 448//14//1 = **32×32 = 1024 tokens/frame**
- KV cache: max_tokens=11264 → max safe frames = (11264−600)//1024 = **10 frames**
- Auto-clamp `num_plm_frames` to avoid KV-cache overflow CUDA crash

---

## pe_description_gen.py — Architecture Decisions

### Modes
- **Default**: draws colored bbox on frames, passes full frame to PLM
- **`--attn-bias`**: no bbox drawing; monkey-patches `F.scaled_dot_product_attention` to suppress non-bbox patches
- **`--compare`**: runs both modes side-by-side, renders split overlay

### Attention Bias
- Fires on **both** prefill (sq==seq_len, is_causal=True) AND generation steps (sq==1)
- `bbox_bias=10.0` → e^-10 ≈ 22000× suppression on non-bbox patches
- `bbox_expand=1.5` → expand tracking bbox 1.5× before computing active patches
- Coordinate hint added to AB prompts: `"The subject person is located near pixel (cx, cy)"`
  — critical so the LLM has a language anchor for which person is the subject

### Patch Grid
- Default: 32×32 (1024 tokens/frame) with pooling_ratio=1
- `--no-pool`: bypass projector's avg pooling → stays 32×32
- `--vis-image-size 896 --no-pool` → 64×64 (4096 tokens/frame, max ~2 frames)
- `_n_patches_side = vis_image_size // patch_size // pool_ratio`

### Taxonomy Labels
```python
BODY_STATES   # 15 motion labels: idle_stand, walk, bend, squat, kneel, ...
OBJ_VERBS     # 20 object verbs:  lift, carry, scan, stack, ...
OBJ_NOUNS_CORE # warehouse objects (used in PLM taxonomy prompt)
SOCIAL_TAX    # 7 labels: none, talk, handover, receive, co_manipulate, ...
SAFETY_EVENTS # 8 labels: none, zone_enter, near_miss, fall, ...
```

---

## pe_video_classify.py — Architecture Decisions

### Pipeline
```
video + track JSON
  → window loop (default 6 s)
    → per identity:
        1. crop bbox from each frame  (hard pixel crop × context_scale=1.5)
        2. encode each crop via PE-Core vision encoder
        3. mean-pool frame embeddings → 1 window embedding [1280-d]
        4. cosine similarity vs pre-computed text label embeddings
        5. top-1 label per slot
```

### Zero-Shot Classification Slots
| Slot | Labels | Notes |
|------|--------|-------|
| `body_state` | 15 | motion labels |
| `obj_verb` | 20 | what action on object |
| `obj_noun` | 93 | COCO 79 + 13 warehouse + `none` |
| `social` | 7 | social interaction type |
| `safety_event` | 8 | safety taxonomy |

### Critical Rules for Text Phrases
- **Do NOT include `"person"` in `OBJ_NOUNS`** — every crop is of a person; it will always win
- **`"none"` phrases must be domain-neutral**: `"nothing unusual happening"`, `"empty hands, not touching anything"` — warehouse-specific phrasing causes wrong labels in non-warehouse scenes
- Keep phrases **short and distinctive** — long overlapping phrases blur embedding distances
- `obj_noun` phrases describe the object itself (not "a person handling X") — keeps it orthogonal to `obj_verb`

### Text Embedding Pre-computation
- Run once at startup: `_build_label_embeddings(pe_model, tokenizer, device)`
- Returns `{slot: (label_list, [N_labels, D] tensor)}`
- All embeddings L2-normalised

### CLI Flags
```bash
--model PE-Core-G14-448    # default, best quality
--window-sec 6.0           # window duration
--num-frames 8             # crops sampled per window per identity
--context-scale 1.5        # bbox expansion for crop
--top-k 3                  # top-k labels stored in JSON + shown in overlay
--debug                    # first 1 minute only + pretty JSON
--no-video                 # skip rendering
--pretty                   # pretty-print JSON
```

### Output JSON Schema
```json
{
  "<identity_id>": [{
    "start_frame": 0, "end_frame": 179,
    "start_sec": 0.0, "end_sec": 6.0,
    "n_frames": 8,
    "body_state": "walk",
    "obj_verb": "carry",
    "obj_noun": "box",
    "social": "none",
    "safety_event": "none",
    "top_k": {
      "body_state":   [["walk", 0.58], ["walk_loaded", 0.41], ...],
      "obj_verb":     [...],
      "obj_noun":     [...],
      "social":       [...],
      "safety_event": [...]
    }
  }]
}
```

---

## Visualization Conventions

Overlay rows drawn **bottom → top** (closest to bbox = lowest):
```
BS:{body_state}  [top-k scores if --top-k>1]   ← identity colour
OBJ:{verb}→{noun}                               ← darker
SC:{social}                                     ← darker still
SE:{safety_event}                               ← dark burgundy (55,45,45)
┌[identity_id]──── bbox ──────────────────────┐
```

Identity colours: deterministic HSV hash of identity string → BGR.
`_identity_color(ident)` and `_darken(color, factor)` are shared helpers.

---

## Track File Format

```json
{
  "<identity_id>": [
    [frame_idx, cx, cy, w, h],
    ...
  ]
}
```
- `cx, cy`: bbox center in pixels
- `w, h`: bbox width and height in pixels
- Entries sorted by frame_idx

---

## Always-Do Rules

1. **`SimpleTokenizer` context_length** — always use `pe_model.context_length`, never hardcode 77
2. **KV-cache guard** — always compute `_max_safe_frames = (max_tokens - 600) // patches_per_frame` and clamp `num_plm_frames`
3. **No `"person"` in obj_noun** — crops are always of a tracked person; it dominates cosine similarity
4. **`"none"` phrases domain-neutral** — never use warehouse-specific language for the "none" fallback label
5. **Commit to `claude/integrate-perception-models-ZWoIC`** — never push to other branches
6. **Push with** `git push -u origin claude/integrate-perception-models-ZWoIC`

---

## PE-Core vs PE-Lang vs PLM — Quick Reference

| | PE-Core | PE-Lang | PLM |
|--|---------|---------|-----|
| Layers | 50 | 47 | 47+LLM |
| pool_type | attn | none | none |
| output_dim | 1280 | None | None |
| Use case | CLIP zero-shot | LLM input features | Generative VQA |
| Text encoder | Yes (CLIP) | No | No (LLM) |
| Reconstruct from each other? | — | No | No |
