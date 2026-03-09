"""
PE Position-Prompted Scene Understanding — Approach 1
Patch Token Selection + Cross-Attention (no retraining of PE required)

Usage (bbox feature extraction):
    python apps/pe/pe_position_approach1.py \
        --image apps/pe/docs/assets/cat.png \
        --bbox 0.1 0.1 0.6 0.6 \
        --model PE-Core-G14-448

Available models:
    PE-Core-G14-448  (width=1536, patch=14, image=448)
    PE-Core-L14-336  (width=1024, patch=14, image=336)
    PE-Core-B16-224  (width=768,  patch=16, image=224)
"""

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# PE model configs (width per model name)
# ---------------------------------------------------------------------------

_PE_WIDTH = {
    "PE-Core-G14-448": 1536,
    "PE-Core-L14-336": 1024,
    "PE-Core-B16-224": 768,
    "PE-Core-S16-384": 384,
    "PE-Core-T16-384": 192,
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class BBoxPrompt:
    """
    Bounding box in normalized image coordinates [0, 1].
        coords = (x1, y1, x2, y2)
    """
    coords: Tuple[float, float, float, float]


# ---------------------------------------------------------------------------
# Patch selection utilities
# ---------------------------------------------------------------------------

def build_patch_grid(image_size: int, patch_size: int) -> torch.Tensor:
    """
    Returns patch center coordinates [N, 2] in [0, 1],
    where N = (image_size // patch_size) ** 2.
    """
    n = image_size // patch_size
    step = 1.0 / n
    coords = torch.arange(n) * step + step / 2.0
    gy, gx = torch.meshgrid(coords, coords, indexing="ij")
    return torch.stack([gx.flatten(), gy.flatten()], dim=-1)  # [N, 2]


def select_bbox_patches(
    bbox: BBoxPrompt,
    patch_grid: torch.Tensor,  # [N, 2]
) -> torch.Tensor:
    """
    Returns indices of patches whose centers fall inside the bounding box.
    Raises ValueError if the box selects no patches.
    """
    x1, y1, x2, y2 = bbox.coords
    mask = (
        (patch_grid[:, 0] >= x1) & (patch_grid[:, 0] <= x2) &
        (patch_grid[:, 1] >= y1) & (patch_grid[:, 1] <= y2)
    )
    indices = mask.nonzero(as_tuple=True)[0]
    if len(indices) == 0:
        raise ValueError(
            f"No patches selected for bbox {bbox.coords}. "
            "Check that coordinates are in [0, 1] and x2 > x1, y2 > y1."
        )
    return indices


# ---------------------------------------------------------------------------
# Cross-attention head (lightweight, trained on top of frozen PE)
# ---------------------------------------------------------------------------

class PositionCrossAttention(nn.Module):
    """
    Cross-attention with a learnable CLS token.

    Query sequence : [CLS, bbox_patch_0, ..., bbox_patch_k]   [B, k+1, D]
    Key/value      : all patch tokens (full image)             [B, N,   D]

    Only the CLS output is returned — it aggregates local bbox context
    from the full image without any manual pooling.
    """

    def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm_q   = nn.LayerNorm(embed_dim)
        self.norm_kv  = nn.LayerNorm(embed_dim)
        self.proj     = nn.Linear(embed_dim, embed_dim)
        self.norm_out = nn.LayerNorm(embed_dim)

    def forward(
        self,
        query_tokens: torch.Tensor,    # [B, k, D]  bbox patches
        context_tokens: torch.Tensor,  # [B, N, D]  full image patches
    ) -> torch.Tensor:                 # [B, D]
        B = query_tokens.shape[0]
        cls = self.cls_token.expand(B, -1, -1)          # [B, 1, D]
        q   = torch.cat([cls, query_tokens], dim=1)     # [B, k+1, D]

        q  = self.norm_q(q)
        kv = self.norm_kv(context_tokens)

        attn_out, _ = self.cross_attn(q, kv, kv)        # [B, k+1, D]
        cls_out = attn_out[:, 0, :]                      # [B, D]
        return self.norm_out(self.proj(cls_out))


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

class PEBBoxFeatureExtractor(nn.Module):
    """
    Frozen PE vision encoder + trainable cross-attention head for
    bounding-box-prompted feature extraction.

    Args:
        pe_vision_encoder : VisionTransformer loaded via
                            ``VisionTransformer.from_config(name, pretrained=True)``
        num_heads         : attention heads in the cross-attention head
    """

    def __init__(
        self,
        pe_vision_encoder: nn.Module,
        num_heads: int = 8,
    ):
        super().__init__()
        self.encoder = pe_vision_encoder

        # Freeze PE
        for p in self.encoder.parameters():
            p.requires_grad_(False)

        embed_dim  = self.encoder.width
        image_size = self.encoder.image_size
        patch_size = self.encoder.patch_size

        self.cross_attn_head = PositionCrossAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
        )

        self.register_buffer(
            "patch_grid",
            build_patch_grid(image_size, patch_size),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _patch_tokens(self, images: torch.Tensor) -> torch.Tensor:
        """
        Run PE and return patch tokens [B, N, width] (no projection).
        Uses forward_features with strip_cls_token=True so the output
        is purely spatial patch tokens regardless of model variant.
        """
        return self.encoder.forward_features(
            images,
            norm=True,
            strip_cls_token=True,
        )  # [B, N, width]

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        images: torch.Tensor,    # [B, C, H, W]
        bboxes: list[BBoxPrompt],
    ) -> torch.Tensor:           # [B, D]
        """Returns L2-normalized bbox-aware feature embeddings."""
        B = images.shape[0]
        assert len(bboxes) == B, "One bbox per image required"

        all_tokens = self._patch_tokens(images)   # [B, N, D]
        D = all_tokens.shape[-1]

        query_list = []
        for b, bbox in enumerate(bboxes):
            idx = select_bbox_patches(bbox, self.patch_grid.to(images.device))
            query_list.append(all_tokens[b, idx])  # [k, D]

        # Pad to uniform k across the batch
        max_k = max(q.shape[0] for q in query_list)
        query_padded = torch.zeros(B, max_k, D, device=images.device)
        for b, q in enumerate(query_list):
            query_padded[b, : q.shape[0]] = q

        embeds = self.cross_attn_head(query_padded, all_tokens)  # [B, D]
        return F.normalize(embeds, dim=-1)

    # ------------------------------------------------------------------
    # Convenience: single image + bbox at inference time
    # ------------------------------------------------------------------

    @torch.no_grad()
    def encode(
        self,
        image: torch.Tensor,  # [C, H, W] or [1, C, H, W]
        bbox: BBoxPrompt,
    ) -> torch.Tensor:        # [D]
        """Returns a single L2-normalized bbox feature vector."""
        self.eval()
        if image.ndim == 3:
            image = image.unsqueeze(0)
        return self.forward(image, [bbox]).squeeze(0)


# ---------------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------------

def load_pe_extractor(
    model_name: str = "PE-Core-G14-448",
    pretrained: bool = True,
    checkpoint_path: Optional[str] = None,
    num_heads: int = 8,
    device: Optional[torch.device] = None,
) -> PEBBoxFeatureExtractor:
    """
    Load a PE vision encoder and wrap it in PEBBoxFeatureExtractor.

    Args:
        model_name      : one of the PE_VISION_CONFIG keys (see config.py)
        pretrained      : download weights from HuggingFace if True
        checkpoint_path : local .pt path; overrides HF download when set
        num_heads       : cross-attention heads (must divide model width)
        device          : target device (defaults to cuda if available)
    """
    from core.vision_encoder.pe import VisionTransformer

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading {model_name} (pretrained={pretrained}) …")
    vision_enc = VisionTransformer.from_config(
        model_name,
        pretrained=pretrained,
        checkpoint_path=checkpoint_path,
    )
    vision_enc = vision_enc.to(device).eval()

    model = PEBBoxFeatureExtractor(vision_enc, num_heads=num_heads).to(device)
    print(
        f"  image_size={vision_enc.image_size}, "
        f"patch_size={vision_enc.patch_size}, "
        f"width={vision_enc.width}"
    )
    return model


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(description="Extract bbox features with PE")
    p.add_argument("--image",  type=str, default=None,
                   help="Path to an image file (PNG/JPEG). "
                        "If omitted a random tensor is used for testing.")
    p.add_argument("--bbox",   type=float, nargs=4,
                   metavar=("X1", "Y1", "X2", "Y2"),
                   default=[0.2, 0.2, 0.7, 0.7],
                   help="Bounding box in normalized [0,1] coords (default: 0.2 0.2 0.7 0.7)")
    p.add_argument("--model",  type=str, default="PE-Core-G14-448",
                   choices=list(_PE_WIDTH.keys()))
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Path to a local .pt checkpoint (skips HF download)")
    p.add_argument("--no-pretrained", action="store_true",
                   help="Skip loading pretrained weights (for quick smoke tests)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pretrained = not args.no_pretrained

    # -- Load model --
    model = load_pe_extractor(
        model_name=args.model,
        pretrained=pretrained,
        checkpoint_path=args.checkpoint,
        device=device,
    )

    # -- Prepare image --
    from core.vision_encoder import transforms as pe_transforms
    preprocess = pe_transforms.get_image_transform(model.encoder.image_size)

    if args.image is not None:
        from PIL import Image as PILImage
        img_tensor = preprocess(PILImage.open(args.image).convert("RGB"))
        print(f"Loaded image: {args.image}")
    else:
        # Random tensor for a smoke test (no real image needed)
        img_tensor = torch.randn(3, model.encoder.image_size, model.encoder.image_size)
        print("No image provided — using random tensor for smoke test")

    img_tensor = img_tensor.to(device)

    # -- Build bbox prompt --
    bbox = BBoxPrompt(coords=tuple(args.bbox))
    print(f"BBox (normalized): x1={bbox.coords[0]}, y1={bbox.coords[1]}, "
          f"x2={bbox.coords[2]}, y2={bbox.coords[3]}")

    # -- Extract feature --
    feat = model.encode(img_tensor, bbox)
    print(f"\nFeature shape : {feat.shape}")
    print(f"L2 norm (≈1.0): {feat.norm().item():.6f}")
    print(f"First 8 values: {feat[:8].tolist()}")
