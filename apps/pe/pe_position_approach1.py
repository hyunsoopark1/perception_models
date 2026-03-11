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

Usage (distillation training from crop.txt):
    python apps/pe/pe_position_approach1.py \
        --train \
        --data-dir /path/to/images \
        --ann-file crop.txt \
        --head-checkpoint head.pt \
        --epochs 10 --lr 1e-4 --batch-size 32

    # crop.txt format (one entry per line):
    #   image_name.jpg  x  y  w  h
    # e.g.: aisol_1x4_2026-01-08_sync000_cam1_000000.jpg 509 189 183 276

Available models:
    PE-Core-G14-448  (width=1536, patch=14, image=448)
    PE-Core-L14-336  (width=1024, patch=14, image=336)
    PE-Core-B16-224  (width=768,  patch=16, image=224)
"""

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image as PILImage

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
# Distillation dataset
# ---------------------------------------------------------------------------

class BBoxDistillationDataset(torch.utils.data.Dataset):
    """
    Loads (image_tensor, BBoxPrompt, crop_tensor) triples for distillation.

    annotations: list of dicts with keys:
        "file_name" : str   path relative to image_dir
        "bbox"      : [x, y, w, h]  top-left + size in pixels

    Accepts COCO-format JSON (with "images"/"annotations" keys) or a plain
    list JSON via :func:`load_distillation_annotations`.
    """

    def __init__(self, image_dir: str, annotations: list,
                 image_transform, crop_transform):
        self.image_dir       = Path(image_dir)
        self.anns            = annotations
        self.image_transform = image_transform
        self.crop_transform  = crop_transform

    def __len__(self) -> int:
        return len(self.anns)

    def __getitem__(self, idx: int):
        ann     = self.anns[idx]
        pil_img = PILImage.open(self.image_dir / ann["file_name"]).convert("RGB")
        img_w, img_h = pil_img.size
        x, y, w, h  = ann["bbox"]
        bbox         = BBoxPrompt(pixel_coords=(x, y, w, h),
                                  image_size=(img_w, img_h))
        crop = bbox.crop(pil_img)
        return self.image_transform(pil_img), bbox, self.crop_transform(crop)


def _collate_distillation(batch):
    """Stack image/crop tensors; keep bboxes as a plain list."""
    imgs, bboxes, crops = zip(*batch)
    return torch.stack(imgs), list(bboxes), torch.stack(crops)


def _collate_cached(batch):
    """Collate pre-computed (patch_tokens, bbox, teacher_embed) triples."""
    patches, bboxes, teachers = zip(*batch)
    return torch.stack(patches), list(bboxes), torch.stack(teachers)


class CachedDataset(torch.utils.data.Dataset):
    """
    Holds pre-computed PE patch tokens and PE teacher embeddings in CPU RAM.
    Eliminates frozen encoder forward passes from the training loop.
    """

    def __init__(
        self,
        patch_tokens: torch.Tensor,    # [n, N, D]
        teacher_embeds: torch.Tensor,  # [n, D]
        bboxes: list,
    ):
        self.patch_tokens   = patch_tokens
        self.teacher_embeds = teacher_embeds
        self.bboxes         = bboxes

    def __len__(self) -> int:
        return len(self.bboxes)

    def __getitem__(self, idx: int):
        return self.patch_tokens[idx], self.bboxes[idx], self.teacher_embeds[idx]


def precompute_features(
    model: "PEBBoxFeatureExtractor",
    pe_model,
    dataloader,
    device: torch.device,
) -> CachedDataset:
    """
    Single pass over the dataset to cache frozen PE patch tokens and PE
    teacher embeddings on CPU.  Training epochs then only run the lightweight
    cross-attention head, which is typically 50-100x cheaper.
    """
    print("Pre-computing frozen patch tokens and teacher embeddings…")
    all_patches: list  = []
    all_teachers: list = []
    all_bboxes: list   = []

    model.eval()
    n_samples = len(dataloader.dataset)
    with torch.no_grad():
        for i, (imgs, bboxes, crops) in enumerate(dataloader):
            imgs  = imgs.to(device)
            crops = crops.to(device)
            patches  = model._patch_tokens(imgs)                      # [B, N, D]
            teachers = pe_model.encode_image(crops, normalize=True)   # [B, D]
            all_patches.append(patches.cpu())
            all_teachers.append(teachers.cpu())
            all_bboxes.extend(bboxes)
            done = min((i + 1) * dataloader.batch_size, n_samples)
            print(f"\r  {done}/{n_samples} samples", end="", flush=True)

    print(f"\r  Cached {len(all_bboxes)} samples.                ")
    return CachedDataset(
        patch_tokens   = torch.cat(all_patches,  dim=0),
        teacher_embeds = torch.cat(all_teachers, dim=0),
        bboxes         = all_bboxes,
    )


def load_distillation_annotations(ann_file: str) -> list:
    """
    Parse an annotation file into a list of {"file_name", "bbox"} dicts.

    Supports:
      - .txt (whitespace-separated): ``image_name.jpg  x  y  w  h``  (one per line)
      - .json plain list : [{"file_name": str, "bbox": [x,y,w,h]}, ...]
      - .json COCO format: {"images": [...], "annotations": [...]}

    Blank lines and lines starting with '#' are ignored in .txt files.
    """
    path = Path(ann_file)
    if path.suffix.lower() == ".txt":
        anns = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) != 5:
                    raise ValueError(
                        f"Expected 'image_name x y w h' but got: {line!r}"
                    )
                name, x, y, w, h = parts
                anns.append({"file_name": name, "bbox": [int(x), int(y), int(w), int(h)]})
        return anns

    # JSON formats
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    # COCO format
    id_to_fn = {img["id"]: img["file_name"] for img in data["images"]}
    return [
        {"file_name": id_to_fn[ann["image_id"]], "bbox": ann["bbox"]}
        for ann in data["annotations"]
        if ann["image_id"] in id_to_fn
    ]


# ---------------------------------------------------------------------------
# Distillation training
# ---------------------------------------------------------------------------

def train_distillation(
    model: "PEBBoxFeatureExtractor",
    pe_model,
    dataloader,
    device: torch.device,
    epochs: int,
    lr: float,
    save_path: Optional[str],
    precompute: bool = True,
    use_amp: bool = True,
) -> None:
    """
    Train PositionCrossAttention via feature distillation.

    Teacher : pe_model.encode_image(crop)   — frozen PE crop embedding
    Student : model(image, bbox)            — cross-attention head output
    Loss    : 1 - cosine_similarity(student, teacher)  averaged over batch

    Speed-ups applied:
      - precompute=True : cache frozen PE tokens + teacher embeds before
                          training; each epoch only runs the tiny head.
      - use_amp=True    : bf16 autocast on CUDA (~1.5-2x faster).
      - pin_memory      : faster CPU→GPU transfers.
      - persistent_workers : avoid worker restart overhead between epochs.
      - zero_grad(set_to_none=True) : skip zeroing, just drop grad tensors.
    """
    use_amp = use_amp and device.type == "cuda"

    if precompute:
        cached_ds = precompute_features(model, pe_model, dataloader, device)
        nw = min(4, dataloader.num_workers)
        train_loader = torch.utils.data.DataLoader(
            cached_ds,
            batch_size=dataloader.batch_size,
            shuffle=True,
            num_workers=nw,
            pin_memory=True,
            persistent_workers=(nw > 0),
            collate_fn=_collate_cached,
        )
    else:
        train_loader = dataloader

    optimizer = torch.optim.AdamW(model.cross_attn_head.parameters(), lr=lr)
    scaler    = torch.cuda.amp.GradScaler(enabled=use_amp)

    for epoch in range(epochs):
        model.cross_attn_head.train()
        if not precompute:
            model.encoder.eval()   # PE stays frozen

        total_loss = 0.0

        if precompute:
            for patches, bboxes, teachers in train_loader:
                patches  = patches.to(device, non_blocking=True)   # [B, N, D]
                teachers = teachers.to(device, non_blocking=True)  # [B, D]

                with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                    enabled=use_amp):
                    student = model._forward_from_patches(patches, bboxes)
                    loss = (1.0 - F.cosine_similarity(student, teachers)).mean()

                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                total_loss += loss.item()
        else:
            for imgs, bboxes, crops in train_loader:
                imgs  = imgs.to(device, non_blocking=True)
                crops = crops.to(device, non_blocking=True)

                with torch.no_grad():
                    teachers = pe_model.encode_image(crops, normalize=True)

                with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                    enabled=use_amp):
                    student = model(imgs, bboxes)
                    loss = (1.0 - F.cosine_similarity(student, teachers)).mean()

                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs}  loss={total_loss / len(train_loader):.6f}")

    if save_path is not None:
        torch.save(model.cross_attn_head.state_dict(), save_path)
        print(f"Head checkpoint saved → {save_path}")


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

    def _forward_from_patches(
        self,
        all_tokens: torch.Tensor,  # [B, N, D]  pre-computed patch tokens
        bboxes: list,
    ) -> torch.Tensor:             # [B, output_dim]
        """
        Head-only forward used during cached training.
        Skips the frozen PE encoder — patch tokens are already available.
        """
        B, _, D = all_tokens.shape
        device  = all_tokens.device

        query_list = []
        for b, bbox in enumerate(bboxes):
            idx = select_bbox_patches(bbox, self.patch_grid.to(device))
            query_list.append(all_tokens[b, idx])

        max_k = max(q.shape[0] for q in query_list)
        query_padded = torch.zeros(B, max_k, D, device=device)
        for b, q in enumerate(query_list):
            query_padded[b, : q.shape[0]] = q

        embeds = self.cross_attn_head(query_padded, all_tokens)  # [B, D]

        if hasattr(self.encoder, "proj") and self.encoder.proj is not None:
            embeds = embeds @ self.encoder.proj

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
    head_checkpoint_path: Optional[str] = None,
    num_heads: int = 8,
    device: Optional[torch.device] = None,
    vision_encoder=None,
) -> PEBBoxFeatureExtractor:
    """
    Load a PE vision encoder and wrap it in PEBBoxFeatureExtractor.

    Args:
        model_name           : one of the PE_VISION_CONFIG keys (see config.py)
        pretrained           : download weights from HuggingFace if True
        checkpoint_path      : local .pt path for PE encoder; overrides HF download
        head_checkpoint_path : local .pt path for PositionCrossAttention weights
        num_heads            : attention heads in the cross-attention head
        device               : target device (defaults to cuda if available)
        vision_encoder       : pre-built VisionTransformer; skips loading when set
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

    if head_checkpoint_path is not None:
        state = torch.load(head_checkpoint_path, map_location=device)
        model.cross_attn_head.load_state_dict(state)
        print(f"  Head weights loaded ← {head_checkpoint_path}")

    return model


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(description="Extract bbox features with PE")

    # ---- shared ----
    p.add_argument("--model", type=str, default="PE-Core-G14-448",
                   choices=list(_PE_WIDTH.keys()))
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Path to a local .pt checkpoint for the PE encoder.")
    p.add_argument("--head-checkpoint", type=str, default=None,
                   metavar="PATH",
                   help="Path to save (training) or load (inference) "
                        "PositionCrossAttention weights.")
    p.add_argument("--no-pretrained", action="store_true",
                   help="Skip loading pretrained PE weights (smoke tests).")

    # ---- inference ----
    p.add_argument("--image", type=str, default=None,
                   help="Path to an image file (PNG/JPEG).")
    p.add_argument("--bbox", type=int, nargs=4, default=None,
                   metavar=("X", "Y", "W", "H"),
                   help="Bounding box as top-left + size: X Y W H in pixels.")
    p.add_argument("--image-size", type=int, nargs=2,
                   metavar=("WIDTH", "HEIGHT"), default=None,
                   help="Original image size (required when --image is omitted).")
    p.add_argument("--crop-out", type=str, default=None, metavar="PATH",
                   help="Save the cropped bbox region to this file.")
    p.add_argument("--text", type=str, nargs="+", default=None,
                   metavar="PHRASE",
                   help="Text phrases to compare via cosine similarity.")

    # ---- distillation training ----
    p.add_argument("--train", action="store_true",
                   help="Run distillation training instead of inference.")
    p.add_argument("--data-dir", type=str, default=None, metavar="DIR",
                   help="Root directory containing training images.")
    p.add_argument("--ann-file", type=str, default=None, metavar="FILE",
                   help="Annotation file: .txt with 'image_name x y w h' per line, "
                        "or .json (plain list or COCO format).")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--no-precompute", action="store_true",
                   help="Disable feature pre-computation (use if RAM is limited).")
    p.add_argument("--no-amp", action="store_true",
                   help="Disable automatic mixed precision (AMP/bf16).")

    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    torch.manual_seed(0)

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pretrained = not args.no_pretrained

    from core.vision_encoder import transforms as pe_transforms
    from core.vision_encoder.pe import CLIP

    # PE model is always needed (distillation teacher or text comparison)
    print(f"Loading {args.model} (pretrained={pretrained}) …")
    pe_model = CLIP.from_config(
        args.model,
        pretrained=pretrained,
        checkpoint_path=args.checkpoint,
    ).to(device).eval()

    model = load_pe_extractor(
        model_name=args.model,
        device=device,
        vision_encoder=pe_model.visual,
        head_checkpoint_path=args.head_checkpoint if not args.train else None,
    )

    preprocess = pe_transforms.get_image_transform(model.encoder.image_size)

    # ------------------------------------------------------------------
    # Training mode
    # ------------------------------------------------------------------
    if args.train:
        if args.data_dir is None or args.ann_file is None:
            raise SystemExit("--train requires --data-dir and --ann-file.")

        annotations = load_distillation_annotations(args.ann_file)
        print(f"Training on {len(annotations)} annotations from {args.ann_file}")

        dataset = BBoxDistillationDataset(
            image_dir=args.data_dir,
            annotations=annotations,
            image_transform=preprocess,
            crop_transform=preprocess,
        )
        nw = args.num_workers
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=nw,
            pin_memory=True,
            persistent_workers=(nw > 0),
            collate_fn=_collate_distillation,
        )

        train_distillation(
            model=model,
            pe_model=pe_model,
            dataloader=dataloader,
            device=device,
            epochs=args.epochs,
            lr=args.lr,
            save_path=args.head_checkpoint,
            precompute=not args.no_precompute,
            use_amp=not args.no_amp,
        )
        sys.exit(0)

    # ------------------------------------------------------------------
    # Inference mode
    # ------------------------------------------------------------------
    if args.bbox is None:
        raise SystemExit("--bbox X Y W H is required for inference.")

    if args.image is not None:
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

    # -- Extract head feature --
    feat = model.encode(img_tensor, bbox)
    print(f"\nFeature shape : {feat.shape}")
    print(f"L2 norm (≈1.0): {feat.norm().item():.6f}")
    print(f"First 8 values: {feat[:8].tolist()}")

    # -- PE crop baseline (always computed when image is available) --
    pe_feat = None
    if pil_img is not None:
        crop_pil    = bbox.crop(pil_img)
        crop_tensor = preprocess(crop_pil).unsqueeze(0).to(device)       # [1, C, H, W]
        with torch.no_grad():
            pe_feat = pe_model.encode_image(crop_tensor, normalize=True)  # [1, D]

        head_pe_sim = F.cosine_similarity(feat.unsqueeze(0), pe_feat).item()
        print(f"\nHead ↔ PE crop similarity : {head_pe_sim:+.4f}"
              f"  (distillation loss equiv: {1.0 - head_pe_sim:.4f})")

    # -- Text comparison --
    if args.text is not None:
        from core.vision_encoder.transforms import get_text_tokenizer

        print("\n--- Text Comparison (cosine similarity) ---")

        tokenizer = get_text_tokenizer(pe_model.context_length)
        tokens = tokenizer(args.text).to(device)                          # [T, L]
        with torch.no_grad():
            text_feats = pe_model.encode_text(tokens, normalize=True)     # [T, D]

        head_sims = (feat.unsqueeze(0) @ text_feats.T).squeeze(0)         # [T]

        if pe_feat is not None:
            pe_sims = (pe_feat @ text_feats.T).squeeze(0)                 # [T]
        else:
            pe_sims = torch.zeros(len(args.text), device=device)
            print("  (PE crop baseline skipped — no source image)")

        logit_scale = pe_model.logit_scale.exp().item()
        col_w = max(len(t) for t in args.text) + 2

        print(f"  {'Phrase':<{col_w}}  {'Head sim':>10}  {'PE crop':>10}  {'Head logit':>11}  {'PE logit':>11}")
        print(f"  {'-'*col_w}  {'----------':>10}  {'----------':>10}  {'-----------':>11}  {'-----------':>11}")
        for phrase, h, c in zip(args.text, head_sims.tolist(), pe_sims.tolist()):
            print(
                f"  {phrase:<{col_w}}  {h:>+10.4f}  {c:>+10.4f}"
                f"  {h * logit_scale:>+11.4f}  {c * logit_scale:>+11.4f}"
            )
