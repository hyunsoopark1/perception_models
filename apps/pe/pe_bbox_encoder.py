# Copyright (c) Meta Platforms, Inc. and affiliates.
# SPDX-License-Identifier: Apache-2.0

"""pe_bbox_encoder.py – BBox-grounded feature extractor for Perception Encoder (PE).

Design
------

Given an image and one or more bounding boxes this module produces a feature
vector that is:

  * **Locally grounded** – anchored to the spatial bbox region.
  * **Globally aware**   – attends over *all* patch tokens, so the full image
    context informs the feature (objects nearby, scene category, etc.).
  * **Token-recycling**  – the expensive ViT forward pass runs *once* per
    image; the lightweight cross-attention pooler then runs once per bbox,
    making multi-box queries on the same image very cheap.

Architecture overview::

    Image (B,C,H,W)
      │
      ▼  VisionTransformer.forward_features()   ← run ONCE per image
    all_patch_tokens  (B, N, D)                 ← cache / recycle
      │
      ├─── bbox → soft spatial weights (area overlap)
      │         → weighted mean of in-bbox tokens
      │         → query  (B, 1, D)               ← local spatial anchor
      │
      └─── BBoxCrossAttentionPooler
              Q  = query          (B, 1, D)
              K  = all_tokens     (B, N, D)       ← full-image context
              V  = all_tokens     (B, N, D)
              │
              ▼ residual MLP
           attended (B, D)
              │
              ▼ optional backbone projection
           bbox_feature  (B, output_dim)

Usage
-----
::

    from apps.pe.pe_bbox_encoder import PEBBoxFeatureExtractor

    extractor = PEBBoxFeatureExtractor.from_pretrained("PE-Core-G14-448")

    # ── Encode the image ONCE ───────────────────────────────────────────────
    image = preprocess(pil_img).unsqueeze(0)          # (1, 3, H, W)
    with torch.no_grad():
        tokens, (grid_h, grid_w) = extractor.encode_image(image)

    # ── Query any number of bboxes without re-running the backbone ──────────
    # Bboxes are normalized xyxy in [0, 1].
    bbox_a = torch.tensor([[0.1, 0.2, 0.5, 0.8]])    # (1, 4)
    feat_a = extractor.extract_bbox_feature(tokens, bbox_a, grid_h, grid_w)
    # (1, D)

    bbox_b = torch.tensor([[0.6, 0.1, 0.9, 0.6]])
    feat_b = extractor.extract_bbox_feature(tokens, bbox_b, grid_h, grid_w)
    # (1, D)

    # ── Or batch multiple bboxes for the same image at once ─────────────────
    bboxes = torch.stack([bbox_a, bbox_b], dim=1)     # (1, 2, 4)
    feats  = extractor.extract_bbox_feature(tokens, bboxes, grid_h, grid_w)
    # (1, 2, D)
"""

from __future__ import annotations

from functools import partial
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.vision_encoder.pe import VisionTransformer
from core.vision_encoder.config import PE_VISION_CONFIG, fetch_pe_checkpoint


# ---------------------------------------------------------------------------
# Cross-attention pooler
# ---------------------------------------------------------------------------

class BBoxCrossAttentionPooler(nn.Module):
    """Pool all image patch tokens into one feature vector grounded to a bbox.

    The query (Q) is derived from the weighted mean of the bbox-region patches,
    giving a spatial anchor.  Keys and values (K, V) come from **all** patch
    tokens so the output attends to the full scene while remaining focused on
    the bbox.  A residual MLP (like a standard transformer block) follows the
    cross-attention step.

    Args:
        embed_dim:   Token dimension D.
        num_heads:   Number of attention heads.
        mlp_ratio:   MLP hidden-dim = embed_dim * mlp_ratio.
        dropout:     Attention dropout probability (default 0 for inference).
        norm_layer:  Normalization layer factory.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        norm_layer=partial(nn.LayerNorm, eps=1e-5),
    ):
        super().__init__()
        assert embed_dim % num_heads == 0, (
            f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
        )

        # Pre-norm for query and key/value streams
        self.q_norm  = norm_layer(embed_dim)
        self.kv_norm = norm_layer(embed_dim)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )

        # Residual MLP following the cross-attention
        mlp_hidden = int(embed_dim * mlp_ratio)
        self.norm2 = norm_layer(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, embed_dim),
        )

    def forward(
        self,
        query: torch.Tensor,       # (B, 1, D)  – bbox region anchor
        all_tokens: torch.Tensor,  # (B, N, D)  – all image patches
    ) -> torch.Tensor:             # (B, D)
        """
        Args:
            query:      Weighted mean of bbox-region patch tokens. Shape (B, 1, D).
            all_tokens: All image patch tokens (full-image context). Shape (B, N, D).

        Returns:
            Aggregated bbox feature. Shape (B, D).
        """
        q  = self.q_norm(query)
        kv = self.kv_norm(all_tokens)

        # Cross-attention: local query, global key/value
        out, _ = self.cross_attn(q, kv, kv, need_weights=False)  # (B, 1, D)

        # Residual MLP
        out = out + self.mlp(self.norm2(out))                     # (B, 1, D)
        return out.squeeze(1)                                      # (B, D)


# ---------------------------------------------------------------------------
# Main extractor
# ---------------------------------------------------------------------------

class PEBBoxFeatureExtractor(nn.Module):
    """BBox-conditioned feature extractor built on a PE VisionTransformer.

    Workflow
    --------
    1. **Encode once** – :meth:`encode_image` runs the full ViT and returns
       all patch tokens.  Wrap in ``torch.no_grad()`` for pure inference.

    2. **Extract per bbox** – :meth:`extract_bbox_feature` uses the cached
       tokens and only runs the lightweight cross-attention pooler, so the
       per-bbox cost is O(N) rather than O(L·N) (L transformer layers).

    Args:
        vision_encoder:   A :class:`~core.vision_encoder.pe.VisionTransformer`
                          instance (pre-built, optionally with weights loaded).
        num_pooler_heads: Attention heads in the cross-attention pooler.
        mlp_ratio:        MLP expansion ratio in the pooler.
        freeze_backbone:  If True the vision encoder weights are frozen.
                          The cross-attention pooler always remains trainable.
    """

    def __init__(
        self,
        vision_encoder: VisionTransformer,
        num_pooler_heads: int = 8,
        mlp_ratio: float = 4.0,
        freeze_backbone: bool = True,
    ):
        super().__init__()
        self.vision_encoder = vision_encoder
        self.patch_size = vision_encoder.patch_size
        embed_dim = vision_encoder.width

        if freeze_backbone:
            self.vision_encoder.requires_grad_(False)

        self.bbox_pooler = BBoxCrossAttentionPooler(
            embed_dim=embed_dim,
            num_heads=num_pooler_heads,
            mlp_ratio=mlp_ratio,
        )

        # Output dimensionality (after optional backbone projection).
        # vision_encoder.output_dim = projected dim if proj exists, else width.
        self.out_dim: int = vision_encoder.output_dim

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _bbox_to_patch_weights(
        bboxes: torch.Tensor,  # (B, 4)  normalized xyxy in [0, 1]
        grid_h: int,
        grid_w: int,
    ) -> torch.Tensor:         # (B, N)  soft weights, sum-to-1 per item
        """Compute soft per-patch weights as fractional area overlap with bbox.

        Each patch cell [c/W, (c+1)/W] × [r/H, (r+1)/H] gets a weight
        proportional to its intersection area with the bbox.  This is
        differentiable w.r.t. bboxes and naturally handles large bboxes
        (uniform weights) and tiny bboxes (spike on the nearest patch).

        For degenerate bboxes with zero overlap (e.g. a point outside the
        grid), the weight is set to 1 on the spatially nearest patch.
        """
        B      = bboxes.shape[0]
        device = bboxes.device
        dtype  = bboxes.dtype

        rows = torch.arange(grid_h, device=device, dtype=dtype)
        cols = torch.arange(grid_w, device=device, dtype=dtype)

        # Patch cell boundaries, broadcast-ready: (1, grid_h, 1) and (1, 1, grid_w)
        py0 = (rows       / grid_h).view(1, grid_h, 1)
        py1 = ((rows + 1) / grid_h).view(1, grid_h, 1)
        px0 = (cols       / grid_w).view(1, 1, grid_w)
        px1 = ((cols + 1) / grid_w).view(1, 1, grid_w)

        # Bbox edges, shape (B, 1, 1)
        bx0 = bboxes[:, 0].view(B, 1, 1)
        by0 = bboxes[:, 1].view(B, 1, 1)
        bx1 = bboxes[:, 2].view(B, 1, 1)
        by1 = bboxes[:, 3].view(B, 1, 1)

        # Per-patch overlap: (B, grid_h, grid_w) → (B, N)
        x_ov  = (torch.minimum(bx1, px1) - torch.maximum(bx0, px0)).clamp(min=0)
        y_ov  = (torch.minimum(by1, py1) - torch.maximum(by0, py0)).clamp(min=0)
        weights = (x_ov * y_ov).reshape(B, -1)  # (B, N)
        total   = weights.sum(dim=1, keepdim=True)

        # Fallback: degenerate bbox with no overlapping patches → nearest patch
        empty = (total.squeeze(1) == 0)
        if empty.any():
            # Patch centers in normalized coords: (N,)
            cy = ((rows + 0.5) / grid_h).view(grid_h, 1).expand(grid_h, grid_w).reshape(-1)
            cx = ((cols + 0.5) / grid_w).view(1, grid_w).expand(grid_h, grid_w).reshape(-1)

            bbox_cx = (bboxes[:, 0] + bboxes[:, 2]) / 2  # (B,)
            bbox_cy = (bboxes[:, 1] + bboxes[:, 3]) / 2

            dist    = (cx.unsqueeze(0) - bbox_cx.unsqueeze(1)) ** 2 \
                    + (cy.unsqueeze(0) - bbox_cy.unsqueeze(1)) ** 2  # (B, N)
            nearest = dist.argmin(dim=1)  # (B,)

            idx = empty.nonzero(as_tuple=True)[0]
            weights[idx, nearest[idx]] = 1.0
            total[idx] = 1.0

        return weights / total  # (B, N), sums to 1

    def _project(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the backbone projection matrix (width → output_dim) if present."""
        if self.vision_encoder.proj_dim is not None:
            # vision_encoder.proj is an nn.Parameter of shape (width, output_dim)
            return x @ self.vision_encoder.proj
        return x

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def encode_image(
        self,
        image: torch.Tensor,  # (B, C, H, W)
    ) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """Run the ViT backbone and return all post-norm patch tokens.

        This is the **token recycling** entry point: call once per image, then
        pass the returned tensors to :meth:`extract_bbox_feature` as many
        times as needed without re-running the backbone.

        Args:
            image: Preprocessed image tensor of shape ``(B, C, H, W)``.

        Returns:
            tokens:  Patch tokens of shape ``(B, N, D)`` where
                     ``N = grid_h * grid_w``.  CLS token is stripped when
                     present (depends on the encoder config).
            grid:    ``(grid_h, grid_w)`` — needed by
                     :meth:`extract_bbox_feature` to map bbox coords to patches.

        Tip:
            Wrap in ``torch.no_grad()`` for pure inference to skip gradient
            bookkeeping::

                with torch.no_grad():
                    tokens, grid = extractor.encode_image(image)
        """
        B, C, H, W = image.shape
        grid_h = H // self.patch_size
        grid_w = W // self.patch_size

        # norm=True  → apply ln_post so features are in the normed space
        # strip_cls_token=True → drop CLS token if the config uses one
        tokens = self.vision_encoder.forward_features(
            image, norm=True, strip_cls_token=True
        )  # (B, N, D)

        return tokens, (grid_h, grid_w)

    def extract_bbox_feature(
        self,
        tokens: torch.Tensor,   # (B, N, D)  – from encode_image()
        bboxes: torch.Tensor,   # (B, 4) or (B, K, 4)  – normalized xyxy
        grid_h: int,
        grid_w: int,
        normalize: bool = False,
    ) -> torch.Tensor:          # (B, D) or (B, K, D)
        """Extract a bbox-grounded feature for each given bounding box.

        Uses the patch tokens cached by :meth:`encode_image`; only the
        lightweight cross-attention pooler runs here.

        Args:
            tokens:    All patch tokens from :meth:`encode_image`. Shape ``(B, N, D)``.
            bboxes:    Bounding boxes in **normalized xyxy** format (values in
                       ``[0, 1]``).  Shape ``(B, 4)`` for one box per image or
                       ``(B, K, 4)`` for K boxes per image.
            grid_h:    Patch grid height (returned by :meth:`encode_image`).
            grid_w:    Patch grid width  (returned by :meth:`encode_image`).
            normalize: If True, L2-normalise the output features.

        Returns:
            Bbox feature(s).  Shape ``(B, D)`` for a single box, or
            ``(B, K, D)`` for K boxes per image.
        """
        multi_box = bboxes.dim() == 3  # True for (B, K, 4) input
        if multi_box:
            B, K, _ = bboxes.shape
            # Flatten to a larger batch so cross-attention runs in one pass.
            bboxes_flat = bboxes.reshape(B * K, 4)
            # Expand tokens: (B, N, D) → (B, K, N, D) → (B*K, N, D)
            tokens_flat = (
                tokens.unsqueeze(1)
                .expand(-1, K, -1, -1)
                .reshape(B * K, tokens.shape[1], tokens.shape[2])
            )
        else:
            bboxes_flat = bboxes   # (B, 4)
            tokens_flat = tokens   # (B, N, D)

        # ── Soft spatial weights ──────────────────────────────────────────────
        # weights[b, n] ∝ overlap area between patch n and bbox b → (B*, N)
        weights = self._bbox_to_patch_weights(bboxes_flat, grid_h, grid_w)

        # ── Weighted query: local spatial anchor from bbox-region patches ─────
        # (B*, N, D) * (B*, N, 1) → sum over N → (B*, 1, D)
        query = (tokens_flat * weights.unsqueeze(-1)).sum(dim=1, keepdim=True)

        # ── Cross-attention pooler: global context informed by local query ─────
        # Q = query (bbox anchor), K/V = all_tokens (full image)  → (B*, D)
        feat = self.bbox_pooler(query, tokens_flat)

        # ── Optional backbone projection (width → output_dim) ─────────────────
        feat = self._project(feat)

        if normalize:
            feat = F.normalize(feat, dim=-1)

        if multi_box:
            feat = feat.reshape(B, K, -1)   # (B, K, D)

        return feat

    def forward(
        self,
        image: torch.Tensor,   # (B, C, H, W)
        bboxes: torch.Tensor,  # (B, 4) or (B, K, 4)  normalized xyxy
        normalize: bool = False,
    ) -> torch.Tensor:
        """End-to-end: image + bboxes → bbox feature(s).

        Prefer :meth:`encode_image` + :meth:`extract_bbox_feature` when the
        same image is queried with multiple bboxes at different times, as this
        avoids redundant backbone passes.

        Args:
            image:     Preprocessed image tensor ``(B, C, H, W)``.
            bboxes:    Normalized xyxy bboxes ``(B, 4)`` or ``(B, K, 4)``.
            normalize: L2-normalise outputs if True.

        Returns:
            ``(B, D)`` or ``(B, K, D)`` bbox feature tensor.
        """
        tokens, (grid_h, grid_w) = self.encode_image(image)
        return self.extract_bbox_feature(
            tokens, bboxes, grid_h, grid_w, normalize=normalize
        )

    @classmethod
    def from_pretrained(
        cls,
        name: str,
        checkpoint_path: Optional[str] = None,
        **kwargs,
    ) -> "PEBBoxFeatureExtractor":
        """Build a ``PEBBoxFeatureExtractor`` from a named PE configuration.

        Downloads and loads the pre-trained backbone weights, then wraps it
        in the extractor with a freshly initialized cross-attention pooler.

        Args:
            name:            PE model name, e.g. ``"PE-Core-G14-448"`` or
                             ``"PE-Spatial-G14-448"``.  Call
                             ``VisionTransformer.available_configs()`` for the
                             full list.
            checkpoint_path: Local path or ``"hf://<repo>:<file>"`` URI.
                             Defaults to the HuggingFace checkpoint for *name*.
            **kwargs:        Forwarded to :class:`PEBBoxFeatureExtractor`
                             (e.g. ``num_pooler_heads``, ``freeze_backbone``).

        Example::

            extractor = PEBBoxFeatureExtractor.from_pretrained(
                "PE-Core-G14-448",
                num_pooler_heads=16,
                freeze_backbone=True,
            )
        """
        if name not in PE_VISION_CONFIG:
            raise ValueError(
                f"Unknown model '{name}'. "
                f"Available: {list(PE_VISION_CONFIG.keys())}"
            )
        vision_encoder = VisionTransformer.from_config(
            name, pretrained=True, checkpoint_path=checkpoint_path
        )
        return cls(vision_encoder, **kwargs)


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    """
    Minimal self-contained smoke-test (no pretrained weights needed).

    Run with:
        python -m apps.pe.pe_bbox_encoder
    """
    import sys

    print("=== PEBBoxFeatureExtractor smoke-test ===\n")

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {DEVICE}")

    # ── Build a tiny encoder (no pretrained weights) ───────────────────────
    from core.vision_encoder.config import PEConfig
    from dataclasses import asdict

    tiny_cfg = PEConfig(
        patch_size=16,
        width=192,       # tiny width for fast test
        layers=2,
        heads=3,
        mlp_ratio=4.0,
        output_dim=256,
        image_size=224,
        pool_type="attn",
        use_cls_token=False,
        use_rope2d=True,
        use_abs_posemb=True,
    )
    vision_encoder = VisionTransformer(**asdict(tiny_cfg)).to(DEVICE)
    extractor = PEBBoxFeatureExtractor(
        vision_encoder, num_pooler_heads=4, freeze_backbone=False
    ).to(DEVICE)

    B, H, W = 2, 224, 224
    image  = torch.randn(B, 3, H, W, device=DEVICE)

    # ── Token recycling demo ───────────────────────────────────────────────
    print("Encoding image (run once) ...")
    tokens, (grid_h, grid_w) = extractor.encode_image(image)
    N, D = tokens.shape[1], tokens.shape[2]
    print(f"  tokens: {tuple(tokens.shape)}  grid: ({grid_h}, {grid_w})")

    # Single bbox per image (B, 4)
    bbox_single = torch.tensor(
        [[0.1, 0.2, 0.5, 0.8],
         [0.3, 0.1, 0.9, 0.6]],
        device=DEVICE,
    )
    feat_single = extractor.extract_bbox_feature(
        tokens, bbox_single, grid_h, grid_w, normalize=True
    )
    print(f"  single bbox feature: {tuple(feat_single.shape)}")
    assert feat_single.shape == (B, extractor.out_dim), "single-box shape mismatch"

    # Multiple bboxes per image (B, K, 4)
    K = 3
    bboxes_multi = torch.rand(B, K, 4, device=DEVICE)
    bboxes_multi[..., 2:] = (bboxes_multi[..., :2] + 0.3).clamp(max=1.0)  # ensure x2>x1
    feat_multi = extractor.extract_bbox_feature(
        tokens, bboxes_multi, grid_h, grid_w, normalize=True
    )
    print(f"  multi  bbox feature: {tuple(feat_multi.shape)}")
    assert feat_multi.shape == (B, K, extractor.out_dim), "multi-box shape mismatch"

    # End-to-end forward (convenience)
    feat_e2e = extractor(image, bbox_single, normalize=True)
    print(f"  end-to-end feature:  {tuple(feat_e2e.shape)}")

    # Degenerate bbox (zero area / point): should not crash
    point_bbox = torch.tensor([[0.5, 0.5, 0.5, 0.5]], device=DEVICE).expand(B, -1)
    feat_pt = extractor.extract_bbox_feature(tokens, point_bbox, grid_h, grid_w)
    print(f"  point bbox feature:  {tuple(feat_pt.shape)}  (degenerate fallback OK)")

    print("\nAll assertions passed. ✓")
    sys.exit(0)
