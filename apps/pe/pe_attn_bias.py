"""
Bbox-guided attention bias for PLM inference.

Background
----------
PLM is LLaVA-style: image patch tokens (CLS stripped, then projected) are
stitched directly into the self-attention token sequence at positions given
by `image_pos` (returned by PLMTokenizer._tokenize_for_generation).

For a 448-px image with patch_size=14 and pooling_ratio=1:
    n_patches_side  = 448 // 14 = 32
    patches_per_frame = 32 × 32 = 1024
    image_pos[f * 1024 + row * 32 + col] = sequence position of patch (f, row, col)

Bias injection
--------------
We monkey-patch torch.nn.functional.scaled_dot_product_attention to add a
[1, 1, 1, seq_len] additive column bias during the prefill pass:

    effective_score[q, k] = (Q[q] · K[k]) / sqrt(d) + bias[k]

Setting bias[k] = BBOX_BIAS for k ∈ bbox-patch positions means every
query token (text + other patches) attends ~e^BBOX_BIAS ≈ 20× more strongly
to the target-person patches.  No model weights are changed.

The bias is automatically skipped for generation steps: during token-by-token
generation the key dimension is the full KV-cache size (larger than seq_len),
so the shape check `S_k == seq_len` fails and the original SDPA runs unmodified.

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
        patches_per_frame=1024, n_patches_side=32,
        orig_w=frame_w, orig_h=frame_h,
        image_size=448, seq_len=seq_len,
        bias=3.0,
    )
    with bbox_attention_bias(mask):
        responses, _, _ = generator.generate([(prompt, frames_tensor)])
"""

import contextlib
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Module-level state for the monkey-patch (one at a time is safe for inference)
# ---------------------------------------------------------------------------

_ACTIVE_BIAS: Optional[torch.Tensor] = None  # shape [1, 1, 1, seq_len]
_ORIG_SDPA = None


def _biased_sdpa(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, **kwargs):
    global _ACTIVE_BIAS
    if _ACTIVE_BIAS is not None:
        sk = key.shape[2]  # [B, H, S, D] layout (after transpose in Attention.forward)
        if _ACTIVE_BIAS.shape[-1] == sk:
            bias = _ACTIVE_BIAS.to(dtype=query.dtype, device=query.device)
            if is_causal:
                # Materialise the causal mask and fold in the bbox bias so
                # we can pass a single attn_mask without is_causal=True.
                sq = query.shape[2]
                causal_mask = torch.zeros(1, 1, sq, sk, dtype=query.dtype, device=query.device)
                causal_mask = causal_mask.masked_fill(
                    torch.ones(sq, sk, device=query.device, dtype=torch.bool).triu(1),
                    float("-inf"),
                )
                attn_mask = causal_mask + bias
                is_causal = False
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
    bias: float = 3.0,
) -> torch.Tensor:
    """
    Build the [1, 1, 1, seq_len] attention bias mask.

    bbox_coords_per_frame : list of (cx, cy, w, h) — one entry per frame
                            (after subsampling).  Length must equal n_frames.
    image_pos             : from get_image_patch_positions().
    patches_per_frame     : e.g. 1024 for 448px / patch_size=14.
    bias                  : additive logit boost for bbox patches.
                            3.0 → ~20× relative attention weight.
    """
    mask = torch.zeros(1, 1, 1, seq_len, dtype=torch.float32)
    for frame_idx, (cx, cy, w, h) in enumerate(bbox_coords_per_frame):
        patch_idxs = bbox_to_patch_indices(
            cx, cy, w, h, orig_w, orig_h, image_size, n_patches_side,
        )
        frame_offset = frame_idx * patches_per_frame
        for p in patch_idxs:
            flat = frame_offset + p
            if flat < len(image_pos):
                seq_pos = image_pos[flat]
                if seq_pos < seq_len:
                    mask[0, 0, 0, seq_pos] = bias
    return mask


@contextlib.contextmanager
def bbox_attention_bias(mask: torch.Tensor):
    """
    Context manager that injects *mask* as an additive attention bias into
    every F.scaled_dot_product_attention call for the duration of the block.

    Only affects prefill (key seq dim == mask seq dim).
    Generation steps (KV-cache dim > mask dim) are automatically skipped.
    """
    global _ACTIVE_BIAS, _ORIG_SDPA
    _ACTIVE_BIAS = mask
    _ORIG_SDPA = F.scaled_dot_product_attention
    F.scaled_dot_product_attention = _biased_sdpa
    try:
        yield
    finally:
        F.scaled_dot_product_attention = _ORIG_SDPA
        _ORIG_SDPA = None
        _ACTIVE_BIAS = None
