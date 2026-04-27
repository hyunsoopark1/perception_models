"""
Bbox-guided attention bias for PLM inference.

Background
----------
PLM is LLaVA-style: image patch tokens (CLS stripped, then projected) are
stitched directly into the self-attention token sequence at positions given
by `image_pos` (returned by PLMTokenizer._tokenize_for_generation).

For a 448-px image with patch_size=14 and pooling_ratio=2 (default):
    n_patches_side   = 448 // 14 // 2 = 16
    patches_per_frame = 16 × 16 = 256
    image_pos[f * 256 + row * 16 + col] = sequence position of patch (f, row, col)

With --no-pool (pooling_ratio=1):
    n_patches_side   = 448 // 14 = 32
    patches_per_frame = 32 × 32 = 1024

Bias injection
--------------
We monkey-patch torch.nn.functional.scaled_dot_product_attention to add a
[1, 1, 1, seq_len] additive column bias during the prefill pass:

    effective_score[q, k] = (Q[q] · K[k]) / sqrt(d) + bias[k]

The mask has two values:
    expanded-bbox patches  →  0      (no bias; compete normally)
    all other image patches →  -bias  (suppressed; e^-10 ≈ 22000× less attention)

The bbox is expanded by bbox_expand (default 1.5×) before selecting active patches.
This gives the model full-body context (needed for pose/motion estimation) while
still hard-excluding other people farther away.  A raw Q·K score of ≥5 is common
for salient distractors (walking adults, etc.), so bias ≥ 10 is required for
reliable person isolation.

No model weights are changed.

Prefill + generation guard
--------------------------
The bias fires during BOTH prefill and generation:
  - Prefill:    sq = len(text_ids) = mask_len   → bias applied (is_causal=True path)
  - Generation: sq = 1                          → bias applied (boolean mask path)
  - Other:      sq ∉ {mask_len, 1}              → skipped (unused in practice)

Applying during generation is essential: non-bbox patch keys remain in the KV
cache after prefill and carry visual features from the initial ViT embeddings.
Without generation suppression, each newly generated query token freely attends
to those keys — causing the model to describe the wrong person even when the
prefill bias was extreme (e.g. -100).

KV-cache note: KVCache.update() returns the full pre-allocated buffer (max_tokens,
e.g. 11264), so sk >> seq_len even during prefill.  The bias is zero-padded from
seq_len to sk so zero-filled cache slots receive neutral (0) bias.

Token-sequence layout (plm_sft template)
-----------------------------------------
BOS + system + <|eot_id|> + user-header
+ <|image|> × (n_frames × patches_per_frame)   ← image tokens come first
+ question text + <|eot_id|> + assistant-header

image_pos[f * patches_per_frame + row * n_patches_side + col]
    = sequence position of patch at (frame f, row, col) in PLM patch grid.
Patches are row-major (left→right, top→bottom), matching ViT output order.

Usage
-----
    from apps.pe.pe_attn_bias import (
        compute_bbox_bias_mask,
        bbox_attention_bias,
        get_image_patch_positions,
    )

    image_pos, seq_len = get_image_patch_positions(generator.tokenizer, prompt, frames_tensor)
    mask = compute_bbox_bias_mask(
        image_pos, bbox_coords_per_frame,
        patches_per_frame=256, n_patches_side=16,
        orig_w=frame_w, orig_h=frame_h,
        image_size=448, seq_len=seq_len,
        bias=10.0, bbox_expand=1.5,
    )
    with bbox_attention_bias(mask):
        responses, _, _ = generator.generate([(prompt, frames_tensor)])
"""

import contextlib
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Module-level state for the monkey-patch (one at a time is safe for inference)
# ---------------------------------------------------------------------------

_ACTIVE_BIAS: Optional[torch.Tensor] = None  # shape [1, 1, 1, seq_len]
_ORIG_SDPA = None
_bias_applied_count = 0   # incremented each time the bias actually fires (for diagnostics)


def _biased_sdpa(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, **kwargs):
    global _ACTIVE_BIAS
    if _ACTIVE_BIAS is not None:
        sq = query.shape[2]   # [B, H, S, D] layout
        sk = key.shape[2]
        mask_len = _ACTIVE_BIAS.shape[-1]

        # Apply during BOTH prefill (sq == mask_len) AND generation (sq == 1).
        #
        # Why generation matters:
        #   During prefill the bias prevents text/bbox tokens from attending to
        #   non-bbox (e.g. teacher) patches.  But those patch keys remain in the
        #   KV cache, and during token-by-token generation there is NO bias —
        #   every newly generated query token can freely attend to the teacher's
        #   cached keys, which still carry visual features from the initial ViT
        #   embeddings.  Result: the model describes the teacher even when the
        #   prefill bias was extreme (e.g. -100).
        #   Applying the suppression during generation closes this loophole.
        if sq == mask_len or sq == 1:
            global _bias_applied_count
            _bias_applied_count += 1
            bias = _ACTIVE_BIAS.to(dtype=query.dtype, device=query.device)

            # Pad bias to full key-cache size.
            # Positions 0..mask_len-1 : prompt tokens (bbox/text bias already set).
            # Positions mask_len..sk-1: generated tokens + cache tail → 0 (no suppression).
            if sk > mask_len:
                bias = torch.cat(
                    [bias, bias.new_zeros(1, 1, 1, sk - mask_len)], dim=-1
                )  # [1, 1, 1, sk]

            if is_causal:
                # Prefill path: materialise standard lower-triangular causal mask
                # (j > i → -inf) and fold in the bbox bias.
                causal_mask = torch.zeros(1, 1, sq, sk, dtype=query.dtype, device=query.device)
                causal_mask.masked_fill_(
                    torch.ones(sq, sk, dtype=torch.bool, device=query.device).triu(1),
                    float("-inf"),
                )
                attn_mask = causal_mask + bias
                is_causal = False
            elif attn_mask is not None and attn_mask.dtype == torch.bool:
                # Generation path: attn_mask is the boolean doc×causal mask
                # [n_seqs, max_tokens] produced by generate_next_token().
                # Convert: True (attend) → 0.0, False (block) → -inf, then add bias.
                float_mask = torch.zeros_like(attn_mask, dtype=bias.dtype)
                float_mask.masked_fill_(~attn_mask, float("-inf"))
                # unsqueeze to [1, 1, n_seqs, sk] for broadcast with bias [1,1,1,sk]
                attn_mask = float_mask.unsqueeze(0).unsqueeze(0) + bias
            else:
                attn_mask = bias if attn_mask is None else attn_mask + bias

    return _ORIG_SDPA(query, key, value,
                      attn_mask=attn_mask, dropout_p=dropout_p,
                      is_causal=is_causal, **kwargs)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_image_patch_positions(
    tokenizer,
    prompt: str,
    frames_tensor: torch.Tensor,
) -> Tuple[List[int], int]:
    """
    Tokenise *prompt* + *frames_tensor* and return (image_pos, seq_len).

    image_pos : flat list of token-sequence positions for every image patch,
                in frame-major order.  Length = n_frames * patches_per_frame.
    seq_len   : total prompt token count (without generation tokens).
    """
    text_ids, image_pos = tokenizer._tokenize_for_generation(prompt, frames_tensor)
    return image_pos, len(text_ids)


def bbox_to_patch_indices(
    cx: float, cy: float, w: float, h: float,
    orig_w: int, orig_h: int,
    image_size: int,
    n_patches_side: int,
) -> List[int]:
    """
    Map a bbox (cx, cy, w, h) in *original* frame pixel coordinates to a list
    of 1-D patch indices (row-major) in the PLM patch grid.

    Parameters
    ----------
    cx, cy, w, h  : bbox centre + size in original-frame pixels.
    orig_w/orig_h : original frame dimensions (before PLM resize).
    image_size    : PLM resize target (e.g. 448).
    n_patches_side: patches per spatial dimension (image_size // patch_size).

    Returns
    -------
    List of patch indices that overlap the rescaled bbox.
    """
    patch_size = image_size / n_patches_side   # float ok here

    # Scale bbox to PLM image coordinates
    sx = image_size / orig_w
    sy = image_size / orig_h
    x1 = max(0.0,            (cx - w / 2) * sx)
    y1 = max(0.0,            (cy - h / 2) * sy)
    x2 = min(float(image_size), (cx + w / 2) * sx)
    y2 = min(float(image_size), (cy + h / 2) * sy)

    col_start = int(x1 / patch_size)
    row_start = int(y1 / patch_size)
    col_end   = min(n_patches_side - 1, int((x2 - 1e-6) / patch_size))
    row_end   = min(n_patches_side - 1, int((y2 - 1e-6) / patch_size))

    indices = []
    for row in range(row_start, row_end + 1):
        for col in range(col_start, col_end + 1):
            indices.append(row * n_patches_side + col)
    return indices


def compute_bbox_bias_mask(
    image_pos: List[int],
    bbox_coords_per_frame: List[Tuple[float, float, float, float]],
    patches_per_frame: int,
    n_patches_side: int,
    orig_w: int,
    orig_h: int,
    image_size: int,
    seq_len: int,
    bias: float = 10.0,
    bbox_expand: float = 1.5,
) -> torch.Tensor:
    """
    Build the [1, 1, 1, seq_len] attention bias mask.

    Mask values
    -----------
    text tokens              : 0      (unmodified)
    expanded-bbox patches    : 0      (unmodified — the model's "viewing region")
    all other image patches  : -bias  (suppressed; e^-10 ≈ 22000x exclusion)

    bbox_expand controls how much the tight tracking bbox is grown before
    computing the active patch set.  1.5 (default) expands width and height
    by 50%, giving the model full-body context and a small margin of
    surrounding scene — enough for pose estimation — while still hard-excluding
    other people standing/walking far from the target.

    With bias=10 (default) the exclusion is near-total (e^-10 ≈ 0.00005).
    For a highly salient distractor (e.g. a walking adult) even bias=5 is
    insufficient because raw Q·K scores can exceed 5; use 10+ to be safe.

    bbox_coords_per_frame : list of (cx, cy, w, h) — one entry per sampled frame.
    bias                  : suppression magnitude for non-active patches.
    bbox_expand           : multiplicative expansion of w and h (default 1.5).
    """
    mask = torch.zeros(1, 1, 1, seq_len, dtype=torch.float32)

    # 1. Suppress all image patch positions
    for pos in image_pos:
        if pos < seq_len:
            mask[0, 0, 0, pos] = -bias

    # 2. Restore expanded-bbox patch positions to 0 (un-suppress target person)
    for frame_idx, (cx, cy, w, h) in enumerate(bbox_coords_per_frame):
        patch_idxs = bbox_to_patch_indices(
            cx, cy, w * bbox_expand, h * bbox_expand,
            orig_w, orig_h, image_size, n_patches_side,
        )
        frame_offset = frame_idx * patches_per_frame
        for p in patch_idxs:
            flat = frame_offset + p
            if flat < len(image_pos):
                seq_pos = image_pos[flat]
                if seq_pos < seq_len:
                    mask[0, 0, 0, seq_pos] = 0.0

    return mask


def make_attn_bias_debug_image(
    frame_rgb: np.ndarray,
    cx: float, cy: float, w: float, h: float,
    orig_w: int, orig_h: int,
    image_size: int = 448,
    n_patches_side: int = 32,
    bias_color: Tuple[int, int, int] = (0, 220, 100),
    alpha: float = 0.45,
    draw_grid: bool = True,
) -> np.ndarray:
    """
    Return a copy of *frame_rgb* (H×W×3 uint8) with the attention-bias
    patch overlay drawn on it.

    Layout
    ------
    • Biased patches  — solid fill at *bias_color* blended with *alpha*.
    • Patch grid      — faint grey lines at every patch boundary (optional).
                        Grid is n_patches_side × n_patches_side (e.g. 16×16
                        when pooling_ratio=2, or 32×32 when pooling_ratio=1).
    • Bbox outline    — solid *bias_color* rectangle, 2 px thick.
    • Patch count     — number of boosted tokens shown at bottom-left.

    Coordinate chain
    ----------------
    original pixels  →  ×(image_size/orig_dim)  →  PLM 448-px space
    patch grid cells    (image_size/n_patches_side) px each in PLM space
    overlay patches  →  ×(orig_dim/image_size)  →  original pixels
    """
    import cv2

    H, W = frame_rgb.shape[:2]
    patch_size_f = image_size / n_patches_side      # 14.0 in the 448-px space
    sx = W / image_size                             # scale patch→original x
    sy = H / image_size                             # scale patch→original y

    bbox_patch_idxs = set(bbox_to_patch_indices(
        cx, cy, w, h, orig_w, orig_h, image_size, n_patches_side,
    ))
    all_patch_idxs = set(range(n_patches_side * n_patches_side))
    suppressed_idxs = all_patch_idxs - bbox_patch_idxs

    # --- alpha-blend: green on bbox patches, dark red on suppressed patches ---
    overlay = frame_rgb.astype(np.float32).copy()
    for p in bbox_patch_idxs:
        row = p // n_patches_side
        col = p % n_patches_side
        ix1 = max(0, int(col * patch_size_f * sx))
        iy1 = max(0, int(row * patch_size_f * sy))
        ix2 = min(W, int((col + 1) * patch_size_f * sx))
        iy2 = min(H, int((row + 1) * patch_size_f * sy))
        overlay[iy1:iy2, ix1:ix2] = bias_color          # boosted (green)
    for p in suppressed_idxs:
        row = p // n_patches_side
        col = p % n_patches_side
        ix1 = max(0, int(col * patch_size_f * sx))
        iy1 = max(0, int(row * patch_size_f * sy))
        ix2 = min(W, int((col + 1) * patch_size_f * sx))
        iy2 = min(H, int((row + 1) * patch_size_f * sy))
        overlay[iy1:iy2, ix1:ix2] = (80, 0, 0)          # suppressed (dark red)

    out = (frame_rgb.astype(np.float32) * (1 - alpha) + overlay * alpha)
    out = out.clip(0, 255).astype(np.uint8)

    # --- faint patch grid ---
    # out is RGB; cv2 drawing writes tuple values directly into the array
    if draw_grid:
        for i in range(1, n_patches_side):
            gx = int(i * patch_size_f * sx)
            gy = int(i * patch_size_f * sy)
            cv2.line(out, (gx, 0), (gx, H - 1), (60, 60, 60), 1, cv2.LINE_AA)
            cv2.line(out, (0, gy), (W - 1, gy), (60, 60, 60), 1, cv2.LINE_AA)

    # --- bbox outline ---
    bx1 = max(0, int(cx - w / 2));  by1 = max(0, int(cy - h / 2))
    bx2 = min(W, int(cx + w / 2));  by2 = min(H, int(cy + h / 2))
    cv2.rectangle(out, (bx1, by1), (bx2, by2), bias_color, 2)

    # --- patch-count label ---
    label = f"boosted={len(bbox_patch_idxs)}  suppressed={len(suppressed_idxs)}"
    cv2.putText(out, label, (6, H - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, bias_color, 1, cv2.LINE_AA)

    return out


@contextlib.contextmanager
def bbox_attention_bias(mask: torch.Tensor):
    """
    Context manager that injects *mask* as an additive attention bias into
    every F.scaled_dot_product_attention call for the duration of the block.

    Only affects prefill (key seq dim == mask seq dim).
    Generation steps (KV-cache dim > mask dim) are automatically skipped.
    """
    global _ACTIVE_BIAS, _ORIG_SDPA, _bias_applied_count
    _ACTIVE_BIAS = mask
    _ORIG_SDPA = F.scaled_dot_product_attention
    F.scaled_dot_product_attention = _biased_sdpa
    _bias_applied_count = 0
    n_suppressed = int((mask < 0).sum())
    n_active = mask.shape[-1] - n_suppressed  # text tokens + bbox patches (bias=0)
    try:
        yield
    finally:
        F.scaled_dot_product_attention = _ORIG_SDPA
        _ORIG_SDPA = None
        _ACTIVE_BIAS = None
        # _bias_applied_count = n_layers × (1 prefill-pass + n_gen_steps)
        print(f"        [AB-dbg] bias fired {_bias_applied_count}x (prefill+gen)  "
              f"active={n_active} suppressed={n_suppressed} / {mask.shape[-1]} prompt tokens")
