"""
PE Position-Prompted Scene Understanding — Approach 1
Patch Token Selection + Cross-Attention (no retraining of PE required)

Usage (bbox feature extraction):
    python apps/pe/pe_position_approach1.py \
        --image apps/pe/docs/assets/cat.png \
        --bbox 50 30 350 270 \
        --model PE-Core-G14-448

    # Without an image (smoke test), supply original image dimensions:
    python apps/pe/pe_position_approach1.py \
        --bbox 50 30 350 270 \
        --image-size 640 480 \
        --no-pretrained

    # --bbox takes X Y W H (top-left corner + width + height)

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
    Bounding box in pixel coordinates of the original (pre-resize) image.

        pixel_coords  = (x, y, w, h)    top-left corner + width/height in pixels
        image_size    = (width, height) of the original image

    Internally converts to (x1, y1, x2, y2) for patch selection and cropping.
    Call `.normalized()` to get (x1, y1, x2, y2) in [0, 1].
    """
    pixel_coords: Tuple[int, int, int, int]   # (x, y, w, h) in pixels
    image_size:   Tuple[int, int]             # (width, height) of source image

    def __post_init__(self):
        x, y, w, h = self.pixel_coords
        # Store as (x1, y1, x2, y2) internally; abs() tolerates negative w/h
        self.pixel_coords = (x, y, x + abs(w), y + abs(h))

    def normalized(self) -> Tuple[float, float, float, float]:
        """Return (x1, y1, x2, y2) normalized to [0, 1]."""
        x1, y1, x2, y2 = self.pixel_coords
        w, h = self.image_size
        return x1 / w, y1 / h, x2 / w, y2 / h

    def crop(self, pil_image):
        """Return the bbox region cropped from a PIL image."""
        x1, y1, x2, y2 = self.pixel_coords
        return pil_image.crop((x1, y1, x2, y2))


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
    x1, y1, x2, y2 = bbox.normalized()
    mask = (
        (patch_grid[:, 0] >= x1) & (patch_grid[:, 0] <= x2) &
        (patch_grid[:, 1] >= y1) & (patch_grid[:, 1] <= y2)
    )
    indices = mask.nonzero(as_tuple=True)[0]
    if len(indices) == 0:
        raise ValueError(
            f"No patches selected for bbox {bbox.pixel_coords} "
            f"(normalized: {bbox.normalized()}). "
            "Check that x2 > x1 and y2 > y1 in pixel coordinates."
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
    ) -> torch.Tensor:           # [B, output_dim]
        """Returns L2-normalized bbox-aware feature embeddings in CLIP output_dim space."""
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

        embeds = self.cross_attn_head(query_padded, all_tokens)  # [B, width]

        # Project into the shared CLIP embedding space (width → output_dim),
        # mirroring VisionTransformer.forward() which applies self.proj after pooling.
        # If no projection exists (proj_dim is None), width == output_dim already.
        if hasattr(self.encoder, "proj") and self.encoder.proj is not None:
            embeds = embeds @ self.encoder.proj                  # [B, output_dim]

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
    vision_encoder=None,
) -> PEBBoxFeatureExtractor:
    """
    Load a PE vision encoder and wrap it in PEBBoxFeatureExtractor.

    Args:
        model_name      : one of the PE_VISION_CONFIG keys (see config.py)
        pretrained      : download weights from HuggingFace if True
        checkpoint_path : local .pt path; overrides HF download when set
        num_heads       : cross-attention heads (must divide model width)
        device          : target device (defaults to cuda if available)
        vision_encoder  : pre-built VisionTransformer; skips loading when set
    """
    from core.vision_encoder.pe import VisionTransformer

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if vision_encoder is None:
        print(f"Loading {model_name} (pretrained={pretrained}) …")
        vision_encoder = VisionTransformer.from_config(
            model_name,
            pretrained=pretrained,
            checkpoint_path=checkpoint_path,
        )
        vision_encoder = vision_encoder.to(device).eval()

    model = PEBBoxFeatureExtractor(vision_encoder, num_heads=num_heads).to(device)
    print(
        f"  image_size={vision_encoder.image_size}, "
        f"patch_size={vision_encoder.patch_size}, "
        f"width={vision_encoder.width}"
    )
    return model


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(description="Extract bbox features with PE")
    p.add_argument("--image", type=str, default=None,
                   help="Path to an image file (PNG/JPEG). "
                        "If omitted, --image-size must be provided.")
    p.add_argument("--bbox", type=int, nargs=4,
                   metavar=("X", "Y", "W", "H"), required=True,
                   help="Bounding box as top-left corner + size: X Y W H in pixels.")
    p.add_argument("--image-size", type=int, nargs=2,
                   metavar=("WIDTH", "HEIGHT"), default=None,
                   help="Original image size in pixels (required when --image is omitted).")
    p.add_argument("--model", type=str, default="PE-Core-G14-448",
                   choices=list(_PE_WIDTH.keys()))
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Path to a local .pt checkpoint (skips HF download)")
    p.add_argument("--no-pretrained", action="store_true",
                   help="Skip loading pretrained weights (for quick smoke tests)")
    p.add_argument("--crop-out", type=str, default=None,
                   metavar="PATH",
                   help="Save the cropped bbox region to this file (PNG/JPEG).")
    p.add_argument("--text", type=str, nargs="+", default=None,
                   metavar="PHRASE",
                   help="One or more text phrases to compare with the bbox region "
                        "via cosine similarity (requires --image).")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pretrained = not args.no_pretrained

    # -- Load model (CLIP when text comparison is requested, VisionTransformer otherwise) --
    clip_model = None
    if args.text is not None:
        from core.vision_encoder.pe import CLIP
        print(f"Loading {args.model} (pretrained={pretrained}) …")
        clip_model = CLIP.from_config(
            args.model,
            pretrained=pretrained,
            checkpoint_path=args.checkpoint,
        ).to(device).eval()
        model = load_pe_extractor(
            model_name=args.model,
            device=device,
            vision_encoder=clip_model.visual,
        )
    else:
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
        pil_img = PILImage.open(args.image).convert("RGB")
        img_w, img_h = pil_img.size
        img_tensor = preprocess(pil_img)
        print(f"Loaded image: {args.image}  ({img_w}x{img_h} px)")
    else:
        pil_img = None
        if args.image_size is None:
            raise SystemExit("--image-size WIDTH HEIGHT is required when --image is omitted.")
        img_w, img_h = args.image_size
        img_tensor = torch.randn(3, model.encoder.image_size, model.encoder.image_size)
        print(f"No image provided — using random tensor (declared size: {img_w}x{img_h} px)")

    img_tensor = img_tensor.to(device)

    # -- Build bbox prompt --
    x, y, w, h = args.bbox
    bbox = BBoxPrompt(pixel_coords=(x, y, w, h), image_size=(img_w, img_h))
    x1, y1, x2, y2 = bbox.pixel_coords
    nx1, ny1, nx2, ny2 = bbox.normalized()
    print(f"BBox (xywh)      : x={x}, y={y}, w={w}, h={h}")
    print(f"BBox (xyxy)      : x1={x1}, y1={y1}, x2={x2}, y2={y2}")
    print(f"BBox (normalized): x1={nx1:.4f}, y1={ny1:.4f}, x2={nx2:.4f}, y2={ny2:.4f}")

    # -- Crop and save bbox region --
    if args.crop_out is not None:
        if pil_img is None:
            print("Warning: --crop-out skipped (no source image provided).")
        else:
            crop = bbox.crop(pil_img)
            crop.save(args.crop_out)
            print(f"Crop saved : {args.crop_out}  ({crop.width}x{crop.height} px)")

    # -- Extract feature --
    feat = model.encode(img_tensor, bbox)
    print(f"\nFeature shape : {feat.shape}")
    print(f"L2 norm (≈1.0): {feat.norm().item():.6f}")
    print(f"First 8 values: {feat[:8].tolist()}")

    # -- Text comparison --
    if args.text is not None:
        from core.vision_encoder.transforms import get_text_tokenizer

        print("\n--- Text Comparison (cosine similarity) ---")

        # feat is already in the shared CLIP output_dim space (via encoder.proj).
        # Tokenize and encode all text phrases into the same space.
        tokenizer = get_text_tokenizer(clip_model.context_length)
        tokens = tokenizer(args.text).to(device)                        # [T, L]
        with torch.no_grad():
            text_feats = clip_model.encode_text(tokens, normalize=True) # [T, output_dim]

        # Cosine similarity: feat · text_feats^T  (both L2-normalised)
        sims = (feat.unsqueeze(0) @ text_feats.T).squeeze(0)            # [T]
        logit_scale = clip_model.logit_scale.exp().item()

        col_w = max(len(t) for t in args.text) + 2
        print(f"  {'Phrase':<{col_w}}  cosine sim   logit-scaled")
        print(f"  {'-'*col_w}  ----------   ------------")
        for phrase, sim in zip(args.text, sims.tolist()):
            print(f"  {phrase:<{col_w}}  {sim:+.4f}       {sim * logit_scale:+.4f}")
