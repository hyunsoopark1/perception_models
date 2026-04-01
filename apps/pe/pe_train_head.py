"""
PE Head Trainer
===============
Distillation training for PositionCrossAttention — the lightweight head that
produces bbox-aware embeddings from frozen PE patch tokens.

How bbox information enters the head (no positional encoding)
-------------------------------------------------------------
The bbox is never fed as a (x, y, w, h) vector.  It enters *implicitly*
through patch selection:

    select_bbox_patches(bbox, patch_grid)
        → patch indices whose centers fall inside the bbox
        → those tokens become the "query" for the cross-attention head

    Stage 1 (self-attn):  [CLS, bbox_patch_0, …, bbox_patch_k]
                           CLS aggregates local bbox-patch information

    Stage 2 (cross-attn): bbox-aware CLS ──► all N patch tokens (full frame)

    Output: CLS token → L2-normalized embedding

Training objective (distillation)
----------------------------------
    Teacher : CLIP.encode_image(crop)     — frozen PE crop embedding
    Student : head(patch_tokens, bbox)    — cross-attention head output
    Loss    : 1 - cosine_similarity(student, teacher)

The frozen PE encoder is run once to cache patch tokens and teacher embeddings;
every training epoch then only runs the tiny cross-attention head (~50-100x
cheaper than running the full encoder each step).

Annotation file format (--ann-file)
-------------------------------------
Plain text — one entry per line::

    image_name.jpg  x  y  w  h

Plain JSON list::

    [{"file_name": "image_name.jpg", "bbox": [x, y, w, h]}, ...]

COCO JSON::

    {"images": [...], "annotations": [...]}

Usage
-----
    python apps/pe/pe_train_head.py \\
        --data-dir  /path/to/images \\
        --ann-file  crop.txt \\
        --model     PE-Core-G14-448 \\
        --out       head.pt

    # Resume from an existing head checkpoint:
    python apps/pe/pe_train_head.py \\
        --data-dir  /path/to/images \\
        --ann-file  crop.txt \\
        --resume    head.pt \\
        --out       head.pt \\
        --epochs    5

    # Fast smoke test (no pretrained weights):
    python apps/pe/pe_train_head.py \\
        --data-dir  /path/to/images \\
        --ann-file  crop.txt \\
        --no-pretrained \\
        --epochs    1 \\
        --out       /tmp/head_test.pt
"""

import argparse
import json
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image as PILImage


# ---------------------------------------------------------------------------
# BBox data structure
# ---------------------------------------------------------------------------

@dataclass
class BBoxPrompt:
    """
    Bounding box in pixel coordinates.

        pixel_coords = (x, y, w, h)  top-left corner + width/height
        image_size   = (width, height) of the source image

    Stores internally as (x1, y1, x2, y2).
    Call .normalized() to get (x1, y1, x2, y2) in [0, 1].
    """
    pixel_coords: Tuple[int, int, int, int]
    image_size:   Tuple[int, int]

    def __post_init__(self):
        x, y, w, h = self.pixel_coords
        self.pixel_coords = (x, y, x + abs(w), y + abs(h))

    def normalized(self) -> Tuple[float, float, float, float]:
        x1, y1, x2, y2 = self.pixel_coords
        w, h = self.image_size
        return x1 / w, y1 / h, x2 / w, y2 / h

    def crop(self, pil_image: PILImage.Image) -> PILImage.Image:
        return pil_image.crop(self.pixel_coords)


# ---------------------------------------------------------------------------
# Patch selection
# ---------------------------------------------------------------------------

def build_patch_grid(image_size: int, patch_size: int) -> torch.Tensor:
    """
    Patch center coordinates [N, 2] in [0, 1].
    N = (image_size // patch_size) ** 2.
    """
    n    = image_size // patch_size
    step = 1.0 / n
    coords = torch.arange(n) * step + step / 2.0
    gy, gx = torch.meshgrid(coords, coords, indexing="ij")
    return torch.stack([gx.flatten(), gy.flatten()], dim=-1)  # [N, 2]


def select_bbox_patches(
    bbox: BBoxPrompt,
    patch_grid: torch.Tensor,   # [N, 2]
) -> torch.Tensor:
    """
    Return indices of patches whose centers fall inside the bbox.

    If no patch center falls inside (e.g. a very thin box at the frame edge),
    return the single nearest patch instead.
    """
    x1, y1, x2, y2 = bbox.normalized()
    x1, y1 = max(0.0, x1), max(0.0, y1)
    x2, y2 = min(1.0, x2), min(1.0, y2)

    mask = (
        (patch_grid[:, 0] >= x1) & (patch_grid[:, 0] <= x2) &
        (patch_grid[:, 1] >= y1) & (patch_grid[:, 1] <= y2)
    )
    indices = mask.nonzero(as_tuple=True)[0]

    if len(indices) == 0:
        cx = torch.tensor([(x1 + x2) / 2.0], device=patch_grid.device)
        cy = torch.tensor([(y1 + y2) / 2.0], device=patch_grid.device)
        center = torch.stack([cx, cy], dim=-1)
        dists  = ((patch_grid - center) ** 2).sum(-1)
        indices = dists.argmin(keepdim=True)

    return indices


# ---------------------------------------------------------------------------
# Cross-attention head
# ---------------------------------------------------------------------------

class PositionCrossAttention(nn.Module):
    """
    Two-stage attention head trained on top of frozen PE patch tokens.

    Bbox information enters through patch selection (not positional encoding):
      - Stage 1 (self-attn):   [CLS, bbox_patches] → CLS aggregates local info
      - Stage 2 (cross-attn):  bbox-aware CLS → all N patch tokens (full frame)

    Only the CLS output is returned.
    """

    def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.self_attn = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads,
            dropout=dropout, batch_first=True,
        )
        self.norm_sa = nn.LayerNorm(embed_dim)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads,
            dropout=dropout, batch_first=True,
        )
        self.norm_q   = nn.LayerNorm(embed_dim)
        self.norm_kv  = nn.LayerNorm(embed_dim)
        self.proj     = nn.Linear(embed_dim, embed_dim)
        self.norm_out = nn.LayerNorm(embed_dim)

    def forward(
        self,
        query_tokens: torch.Tensor,    # [B, k, D]  bbox patches
        context_tokens: torch.Tensor,  # [B, N, D]  all patch tokens
    ) -> torch.Tensor:                 # [B, D]
        B   = query_tokens.shape[0]
        cls = self.cls_token.expand(B, -1, -1)       # [B, 1, D]
        q   = torch.cat([cls, query_tokens], dim=1)  # [B, k+1, D]

        # Stage 1: self-attention — CLS gathers local bbox info
        q_sa = self.norm_sa(q)
        q    = q + self.self_attn(q_sa, q_sa, q_sa)[0]

        # Stage 2: cross-attention — bbox-aware CLS attends to full frame
        q  = self.norm_q(q)
        kv = self.norm_kv(context_tokens)
        attn_out, _ = self.cross_attn(q, kv, kv)    # [B, k+1, D]
        cls_out = attn_out[:, 0, :]                  # [B, D]
        return self.norm_out(self.proj(cls_out))


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class BBoxDistillationDataset(torch.utils.data.Dataset):
    """
    Returns (image_tensor, BBoxPrompt, crop_tensor) for each annotation.

    image_tensor : full image preprocessed for PE encoder
    crop_tensor  : bbox region preprocessed for PE encoder (teacher target)
    """

    def __init__(
        self,
        image_dir: str,
        annotations: List[dict],
        image_transform,
        crop_transform,
    ):
        self.image_dir       = Path(image_dir)
        self.anns            = annotations
        self.image_transform = image_transform
        self.crop_transform  = crop_transform

    def __len__(self) -> int:
        return len(self.anns)

    def __getitem__(self, idx: int):
        ann      = self.anns[idx]
        pil_img  = PILImage.open(self.image_dir / ann["file_name"]).convert("RGB")
        img_w, img_h = pil_img.size
        x, y, w, h   = ann["bbox"]
        bbox = BBoxPrompt(pixel_coords=(x, y, w, h), image_size=(img_w, img_h))
        crop = bbox.crop(pil_img)
        return self.image_transform(pil_img), bbox, self.crop_transform(crop)


def _collate_distillation(batch):
    imgs, bboxes, crops = zip(*batch)
    return torch.stack(imgs), list(bboxes), torch.stack(crops)


def _collate_cached(batch):
    patches, bboxes, teachers = zip(*batch)
    return torch.stack(patches), list(bboxes), torch.stack(teachers)


# ---------------------------------------------------------------------------
# Annotation loader
# ---------------------------------------------------------------------------

def load_annotations(ann_file: str) -> List[dict]:
    """
    Parse annotations into [{"file_name": str, "bbox": [x,y,w,h]}, ...].

    Supports .txt (image_name x y w h per line), plain JSON list, COCO JSON.
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
                    raise ValueError(f"Expected 'image_name x y w h', got: {line!r}")
                name, x, y, w, h = parts
                anns.append({"file_name": name, "bbox": [int(x), int(y), int(w), int(h)]})
        return anns

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
# Feature pre-computation (cache frozen encoder outputs before training)
# ---------------------------------------------------------------------------

class CachedDataset(torch.utils.data.Dataset):
    """Pre-computed patch tokens + teacher embeddings in CPU RAM / disk mmap."""

    def __init__(
        self,
        patch_tokens:   torch.Tensor,   # [n, N, D]
        teacher_embeds: torch.Tensor,   # [n, D]
        bboxes:         List,
    ):
        self.patch_tokens   = patch_tokens
        self.teacher_embeds = teacher_embeds
        self.bboxes         = bboxes

    def __len__(self) -> int:
        return len(self.bboxes)

    def __getitem__(self, idx: int):
        return (
            self.patch_tokens[idx].float(),
            self.bboxes[idx],
            self.teacher_embeds[idx].float(),
        )


def precompute_features(
    vision_encoder: nn.Module,
    pe_model,
    dataloader,
    device: torch.device,
) -> CachedDataset:
    """
    Single pass: cache frozen PE patch tokens and teacher crop embeddings.
    Uses disk-backed float16 memmaps to avoid OOM on large datasets.
    """
    print("Pre-computing frozen patch tokens and teacher embeddings …")

    # Re-create loader with num_workers=0 to avoid CUDA + fork deadlock.
    precompute_loader = torch.utils.data.DataLoader(
        dataloader.dataset,
        batch_size=dataloader.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=dataloader.collate_fn,
    )

    n_samples = len(precompute_loader.dataset)

    # Probe one sample to learn tensor shapes before allocating.
    probe_loader = torch.utils.data.DataLoader(
        dataloader.dataset, batch_size=1, num_workers=0,
        collate_fn=dataloader.collate_fn,
    )
    probe_imgs, _, probe_crops = next(iter(probe_loader))
    with torch.no_grad():
        probe_patches = vision_encoder.forward_features(
            probe_imgs.to(device), norm=True, strip_cls_token=True
        )  # [1, N, D]
        probe_teacher = pe_model.encode_image(
            probe_crops.to(device), normalize=True
        )  # [1, D_out]

    N_patches = probe_patches.shape[1]
    D_patch   = probe_patches.shape[2]
    D_teacher = probe_teacher.shape[1]

    tmp_dir      = Path(tempfile.mkdtemp(prefix="pe_cache_"))
    patches_path  = tmp_dir / "patches.bin"
    teachers_path = tmp_dir / "teachers.bin"
    patches_mm  = np.memmap(patches_path,  dtype="float16", mode="w+",
                             shape=(n_samples, N_patches, D_patch))
    teachers_mm = np.memmap(teachers_path, dtype="float16", mode="w+",
                             shape=(n_samples, D_teacher))

    all_bboxes: List = []
    offset = 0
    with torch.no_grad():
        for imgs, bboxes, crops in precompute_loader:
            imgs  = imgs.to(device)
            crops = crops.to(device)
            try:
                patches = vision_encoder.forward_features(
                    imgs, norm=True, strip_cls_token=True
                )  # [B, N, D]
                teachers = pe_model.encode_image(crops, normalize=True)  # [B, D]
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower():
                    torch.cuda.empty_cache()
                    raise RuntimeError(
                        "CUDA OOM during precomputation — reduce --batch-size"
                    ) from exc
                raise
            B = imgs.shape[0]
            patches_mm[offset : offset + B]  = patches.cpu().half().numpy()
            teachers_mm[offset : offset + B] = teachers.cpu().half().numpy()
            all_bboxes.extend(bboxes)
            offset += B
            print(f"\r  {min(offset, n_samples)}/{n_samples}", end="", flush=True)

    patches_mm.flush()
    teachers_mm.flush()

    patches_ro  = np.memmap(patches_path,  dtype="float16", mode="r",
                             shape=(n_samples, N_patches, D_patch))
    teachers_ro = np.memmap(teachers_path, dtype="float16", mode="r",
                             shape=(n_samples, D_teacher))
    print(f"\r  Cached {len(all_bboxes)} samples to {tmp_dir}")

    return CachedDataset(
        patch_tokens   = torch.from_numpy(patches_ro),
        teacher_embeds = torch.from_numpy(teachers_ro),
        bboxes         = all_bboxes,
    )


# ---------------------------------------------------------------------------
# Head forward helper (also used by inference scripts)
# ---------------------------------------------------------------------------

def head_forward(
    head: PositionCrossAttention,
    patch_tokens: torch.Tensor,   # [B, N, D]  already on device
    bboxes: List[BBoxPrompt],
    patch_grid: torch.Tensor,     # [N, 2]
    proj: Optional[torch.Tensor], # [D, output_dim] or None
    device: torch.device,
) -> torch.Tensor:                # [B, output_dim], normalized
    """
    Run the cross-attention head for a batch of (patch_tokens, bbox) pairs.
    Applies the PE projection matrix if provided.
    """
    B, _, D = patch_tokens.shape

    query_list = []
    for b, bbox in enumerate(bboxes):
        idx = select_bbox_patches(bbox, patch_grid.to(device))
        query_list.append(patch_tokens[b, idx])

    max_k = max(q.shape[0] for q in query_list)
    query_padded = torch.zeros(B, max_k, D, device=device)
    for b, q in enumerate(query_list):
        query_padded[b, : q.shape[0]] = q

    embeds = head(query_padded, patch_tokens)   # [B, D]

    if proj is not None:
        embeds = embeds @ proj                  # [B, output_dim]

    return F.normalize(embeds, dim=-1)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(
    head: PositionCrossAttention,
    vision_encoder: nn.Module,
    pe_model,
    dataloader,
    patch_grid: torch.Tensor,
    proj: Optional[torch.Tensor],
    device: torch.device,
    epochs: int,
    lr: float,
    out_path: str,
    precompute: bool = True,
    use_amp: bool = True,
) -> None:
    """
    Distillation training loop.

    Teacher : pe_model.encode_image(crop)         — frozen PE crop embedding
    Student : head_forward(patch_tokens, bbox)    — cross-attention head
    Loss    : 1 - cosine_similarity(student, teacher)
    """
    use_amp = use_amp and device.type == "cuda"

    if precompute:
        cached_ds = precompute_features(vision_encoder, pe_model, dataloader, device)
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

    optimizer = torch.optim.AdamW(head.parameters(), lr=lr)
    scaler    = torch.cuda.amp.GradScaler(enabled=use_amp)

    for epoch in range(epochs):
        head.train()
        total_loss = 0.0

        if precompute:
            for patch_tokens, bboxes, teachers in train_loader:
                patch_tokens = patch_tokens.to(device, non_blocking=True)  # [B, N, D]
                teachers     = teachers.to(device, non_blocking=True)      # [B, D]

                with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                    enabled=use_amp):
                    student = head_forward(
                        head, patch_tokens, bboxes, patch_grid, proj, device
                    )  # [B, output_dim]
                    loss = (1.0 - F.cosine_similarity(student, teachers)).mean()

                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                total_loss += loss.item()

        else:
            vision_encoder.eval()
            for imgs, bboxes, crops in train_loader:
                imgs  = imgs.to(device, non_blocking=True)
                crops = crops.to(device, non_blocking=True)

                with torch.no_grad():
                    patch_tokens = vision_encoder.forward_features(
                        imgs, norm=True, strip_cls_token=True
                    )  # [B, N, D]
                    teachers = pe_model.encode_image(crops, normalize=True)  # [B, D]

                with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                    enabled=use_amp):
                    student = head_forward(
                        head, patch_tokens, bboxes, patch_grid, proj, device
                    )
                    loss = (1.0 - F.cosine_similarity(student, teachers)).mean()

                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                total_loss += loss.item()

        avg = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{epochs}  loss={avg:.6f}")

    torch.save(head.state_dict(), out_path)
    print(f"Saved → {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(
        description="Distillation training for the PositionCrossAttention head.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--data-dir",  required=True, metavar="DIR",
                   help="Root directory of training images.")
    p.add_argument("--ann-file",  required=True, metavar="FILE",
                   help="Annotation file: .txt (image x y w h), plain JSON, or COCO JSON.")
    p.add_argument("--out",       required=True, metavar="PATH",
                   help="Output path for the trained head checkpoint (.pt).")
    p.add_argument("--model",     default="PE-Core-G14-448",
                   choices=["PE-Core-G14-448", "PE-Core-L14-336",
                            "PE-Core-B16-224", "PE-Core-S16-384", "PE-Core-T16-384"])
    p.add_argument("--checkpoint", default=None, metavar="PATH",
                   help="PE encoder checkpoint (.pt); overrides HuggingFace weights.")
    p.add_argument("--resume",    default=None, metavar="PATH",
                   help="Resume training from an existing head checkpoint (.pt).")
    p.add_argument("--no-pretrained", action="store_true",
                   help="Skip loading pretrained PE weights (smoke test).")
    p.add_argument("--num-heads", type=int, default=8,
                   help="Attention heads in PositionCrossAttention (default: 8).")
    p.add_argument("--epochs",    type=int, default=10)
    p.add_argument("--lr",        type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--no-precompute", action="store_true",
                   help="Disable feature pre-computation (slower, lower RAM).")
    p.add_argument("--no-amp",    action="store_true",
                   help="Disable bf16 autocast.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = _parse_args()
    torch.manual_seed(0)

    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pretrained = not args.no_pretrained

    from core.vision_encoder.pe import CLIP
    from core.vision_encoder.transforms import get_image_transform

    # Load full CLIP model (visual encoder + text encoder used as teacher)
    print(f"Loading {args.model} (pretrained={pretrained}) …")
    pe_model = CLIP.from_config(
        args.model, pretrained=pretrained, checkpoint_path=args.checkpoint
    ).to(device).eval()

    vision_encoder = pe_model.visual   # frozen during training
    for p in vision_encoder.parameters():
        p.requires_grad_(False)

    # Build patch grid and extract projection matrix once
    image_size = vision_encoder.image_size
    patch_size = vision_encoder.patch_size
    width      = vision_encoder.width
    patch_grid = build_patch_grid(image_size, patch_size).to(device)

    proj: Optional[torch.Tensor] = None
    if hasattr(vision_encoder, "proj") and vision_encoder.proj is not None:
        proj = vision_encoder.proj.to(device)

    print(f"  image_size={image_size}  patch_size={patch_size}  width={width}")

    # Build cross-attention head
    head = PositionCrossAttention(embed_dim=width, num_heads=args.num_heads).to(device)
    if args.resume is not None:
        state = torch.load(args.resume, map_location=device, weights_only=True)
        missing, unexpected = head.load_state_dict(state, strict=False)
        if missing:
            print(f"  Warning — missing keys (random init): {missing}")
        if unexpected:
            print(f"  Warning — unexpected keys (ignored): {unexpected}")
        print(f"  Resumed from {args.resume}")

    # Dataset
    preprocess  = get_image_transform(image_size)
    annotations = load_annotations(args.ann_file)
    print(f"Training on {len(annotations)} annotations from {args.ann_file}")

    dataset = BBoxDistillationDataset(
        image_dir        = args.data_dir,
        annotations      = annotations,
        image_transform  = preprocess,
        crop_transform   = preprocess,
    )
    nw = args.num_workers
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size        = args.batch_size,
        shuffle           = True,
        num_workers       = nw,
        pin_memory        = True,
        persistent_workers= (nw > 0),
        collate_fn        = _collate_distillation,
    )

    # Train
    train(
        head           = head,
        vision_encoder = vision_encoder,
        pe_model       = pe_model,
        dataloader     = dataloader,
        patch_grid     = patch_grid,
        proj           = proj,
        device         = device,
        epochs         = args.epochs,
        lr             = args.lr,
        out_path       = args.out,
        precompute     = not args.no_precompute,
        use_amp        = not args.no_amp,
    )
