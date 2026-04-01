"""
PE Head Trainer
===============
Distillation training for PositionCrossAttention.

Input data
----------
A single folder containing paired video and track files:

    data_dir/
        1.MP4   1.json
        2.MP4   2.json
        ...

Track JSON format (same as pe_feature_similarity_viz.py / pe_combined_viz.py):

    {
      "identity_a": [[frame_idx, cx, cy, w, h], ...],
      "identity_b": [[frame_idx, cx, cy, w, h], ...]
    }

cx, cy is the bbox center in pixels; w, h is width/height.
One frame per identity is sampled every --sample-interval seconds (default: 10).

How bbox information enters the head (no positional encoding)
-------------------------------------------------------------
The bbox selects which patch tokens become the cross-attention query:

    select_bbox_patches(bbox, patch_grid)
        → indices of patches whose centers fall inside the bbox
        → those tokens become the query for the head

    Stage 1 (self-attn):  [CLS, bbox_patches]  — CLS absorbs local bbox info
    Stage 2 (cross-attn): bbox-aware CLS ──► all N patch tokens (full frame)
    Output: CLS → L2-normalized embedding

Training objective (distillation)
----------------------------------
    Teacher : CLIP.encode_image(crop)         frozen PE crop embedding
    Student : head(patch_tokens, bbox)        cross-attention head output
    Loss    : 1 - cosine_similarity(student, teacher)

Precomputation
--------------
Before training starts, the script reads every video frame once, computes
frozen patch tokens (once per frame, shared across all bboxes in that frame)
and teacher crop embeddings (once per bbox), and stores them in disk-backed
float16 memmaps.  Each training epoch then only runs the tiny head.

Usage
-----
    python apps/pe/pe_train_head.py \\
        --data-dir  /path/to/videos \\
        --model     PE-Core-G14-448 \\
        --out       head.pt

    # Resume from an existing checkpoint:
    python apps/pe/pe_train_head.py \\
        --data-dir  /path/to/videos \\
        --resume    head.pt \\
        --out       head.pt \\
        --epochs    5
"""

import json
import sys
import tempfile
import argparse
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
    Bounding box in pixel coordinates of the original (pre-resize) image.
    pixel_coords = (x, y, w, h)  top-left corner + width/height.
    Stores internally as (x1, y1, x2, y2).
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


# ---------------------------------------------------------------------------
# Patch selection
# ---------------------------------------------------------------------------

def build_patch_grid(image_size: int, patch_size: int) -> torch.Tensor:
    """Patch center coordinates [N, 2] in [0, 1]."""
    n    = image_size // patch_size
    step = 1.0 / n
    coords = torch.arange(n) * step + step / 2.0
    gy, gx = torch.meshgrid(coords, coords, indexing="ij")
    return torch.stack([gx.flatten(), gy.flatten()], dim=-1)  # [N, 2]


def select_bbox_patches(bbox: BBoxPrompt, patch_grid: torch.Tensor) -> torch.Tensor:
    """
    Indices of patches whose centers fall inside the bbox.
    Falls back to the nearest patch if the box is a sub-patch sliver.
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
        indices = ((patch_grid - center) ** 2).sum(-1).argmin(keepdim=True)

    return indices


# ---------------------------------------------------------------------------
# Cross-attention head
# ---------------------------------------------------------------------------

class PositionCrossAttention(nn.Module):
    """
    Stage 1 (self-attn):  [CLS, bbox_patches] attend to each other.
                           CLS absorbs local bbox content.
    Stage 2 (cross-attn): bbox-aware CLS attends to all N patch tokens.
    Output: CLS token, L2-normalized.
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
        query_tokens:   torch.Tensor,  # [B, k, D]  bbox patches
        context_tokens: torch.Tensor,  # [B, N, D]  all patch tokens
    ) -> torch.Tensor:                 # [B, D]
        B   = query_tokens.shape[0]
        cls = self.cls_token.expand(B, -1, -1)       # [B, 1, D]
        q   = torch.cat([cls, query_tokens], dim=1)  # [B, k+1, D]

        q_sa = self.norm_sa(q)
        q    = q + self.self_attn(q_sa, q_sa, q_sa)[0]

        q  = self.norm_q(q)
        kv = self.norm_kv(context_tokens)
        attn_out, _ = self.cross_attn(q, kv, kv)
        cls_out = attn_out[:, 0, :]
        return self.norm_out(self.proj(cls_out))


# ---------------------------------------------------------------------------
# Head forward helper
# ---------------------------------------------------------------------------

def head_forward(
    head:        PositionCrossAttention,
    patch_tokens: torch.Tensor,    # [B, N, D]
    bboxes:       List[BBoxPrompt],
    patch_grid:   torch.Tensor,    # [N, 2]
    proj:         Optional[torch.Tensor],
    device:       torch.device,
) -> torch.Tensor:                 # [B, output_dim], L2-normalized
    B, _, D = patch_tokens.shape

    query_list = [
        patch_tokens[b, select_bbox_patches(bbox, patch_grid.to(device))]
        for b, bbox in enumerate(bboxes)
    ]
    max_k = max(q.shape[0] for q in query_list)
    query_padded = torch.zeros(B, max_k, D, device=device)
    for b, q in enumerate(query_list):
        query_padded[b, : q.shape[0]] = q

    embeds = head(query_padded, patch_tokens)   # [B, D]
    if proj is not None:
        embeds = embeds @ proj                  # [B, output_dim]
    return F.normalize(embeds, dim=-1)


# ---------------------------------------------------------------------------
# Cached dataset (disk-backed memmaps)
# ---------------------------------------------------------------------------

class CachedDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        patch_tokens:   torch.Tensor,  # [n, N, D]
        teacher_embeds: torch.Tensor,  # [n, D]
        bboxes:         List[BBoxPrompt],
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


def _collate_cached(batch):
    patches, bboxes, teachers = zip(*batch)
    return torch.stack(patches), list(bboxes), torch.stack(teachers)


# ---------------------------------------------------------------------------
# Scan folder for (video, json) pairs and build sample manifest
# ---------------------------------------------------------------------------

VIDEO_EXTENSIONS = {".mp4", ".MP4", ".mov", ".MOV", ".avi", ".AVI"}


def scan_video_folder(
    data_dir:        str,
    sample_interval: float = 10.0,
) -> List[Tuple]:
    """
    Scan data_dir for paired video + JSON track files.

    For each identity, the track entries are grouped into non-overlapping
    windows of length sample_interval seconds.  The middle entry of each
    window is kept.  This yields roughly one sample per identity per
    sample_interval seconds of footage.

    Returns a flat list of samples:
        (video_path, frame_idx, cx, cy, w, h, frame_w, frame_h)
    """
    try:
        import cv2
    except ImportError:
        sys.exit("opencv-python is required:  pip install opencv-python-headless")

    data_path = Path(data_dir)
    pairs = []
    for f in sorted(data_path.iterdir()):
        if f.suffix in VIDEO_EXTENSIONS:
            json_path = f.with_suffix(".json")
            if json_path.exists():
                pairs.append((f, json_path))
            else:
                print(f"  Warning: no JSON for {f.name} — skipped")

    if not pairs:
        sys.exit(f"No (video, JSON) pairs found in {data_dir}")

    print(f"Found {len(pairs)} video/JSON pairs in {data_dir}")

    samples = []
    for video_path, json_path in pairs:
        cap      = cv2.VideoCapture(str(video_path))
        fps      = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        # Number of frames per sampling window
        frame_step = max(1, int(fps * sample_interval))

        with open(json_path) as f:
            data = json.load(f)

        n_before = len(samples)
        for identity, entries in data.items():
            # Group track entries into sample_interval-second windows.
            # Pick the middle entry of each window as the representative sample.
            windows: Dict[int, List] = defaultdict(list)
            for entry in entries:
                wid = int(entry[0]) // frame_step
                windows[wid].append(entry)

            for wid in sorted(windows):
                window_entries = windows[wid]
                mid = window_entries[len(window_entries) // 2]
                fidx, cx, cy, w, h = mid
                samples.append((
                    str(video_path), int(fidx),
                    float(cx), float(cy), float(w), float(h),
                    frame_w, frame_h,
                ))

        duration_s = n_frames / fps
        print(f"  {video_path.name}  "
              f"{duration_s:.0f}s  fps={fps:.1f}  step={frame_step}fr  "
              f"{len(samples) - n_before} samples")

    print(f"Total: {len(samples)} (frame, bbox) samples")
    return samples


# ---------------------------------------------------------------------------
# Precompute patch tokens + teacher embeddings from video frames
# ---------------------------------------------------------------------------

def precompute_from_videos(
    samples:         List[Tuple],
    vision_encoder:  nn.Module,
    pe_model,
    image_transform,
    device:          torch.device,
    context_scale:   float = 1.0,
    batch_size:      int   = 16,
) -> CachedDataset:
    """
    Read video frames sequentially, compute and cache:
      • patch_tokens  [N, D]  — frozen PE patch tokens for the full frame
                                 (computed once per frame, shared across bboxes)
      • teacher_embed [D]     — PE crop embedding for each bbox
                                 (computed once per bbox)

    Results are stored in disk-backed float16 memmaps so large datasets
    don't exhaust RAM.
    """
    try:
        import cv2
    except ImportError:
        sys.exit("opencv-python is required:  pip install opencv-python-headless")

    n_samples = len(samples)
    print(f"\nPrecomputing features for {n_samples} samples …")

    # ── Probe one sample to learn output shapes ───────────────────────
    # Use known model properties — no need to decode a real frame.
    N_patches = (vision_encoder.image_size // vision_encoder.patch_size) ** 2
    D_patch   = vision_encoder.width
    if hasattr(vision_encoder, "proj") and vision_encoder.proj is not None:
        D_teacher = vision_encoder.proj.shape[1]
    else:
        D_teacher = vision_encoder.width

    # ── Allocate disk-backed memmaps (float16 to halve disk/I-O) ─────
    tmp_dir      = Path(tempfile.mkdtemp(prefix="pe_cache_"))
    patches_mm   = np.memmap(tmp_dir / "patches.bin",  dtype="float16", mode="w+",
                              shape=(n_samples, N_patches, D_patch))
    teachers_mm  = np.memmap(tmp_dir / "teachers.bin", dtype="float16", mode="w+",
                              shape=(n_samples, D_teacher))

    # ── Group samples by (video, frame_idx) for sequential reads ─────
    # video_path → {frame_idx → [sample_indices]}
    video_frame_map: Dict[str, Dict[int, List[int]]] = defaultdict(lambda: defaultdict(list))
    for i, (vpath, fidx, *_) in enumerate(samples):
        video_frame_map[vpath][fidx].append(i)

    all_bboxes: List[BBoxPrompt] = [None] * n_samples  # type: ignore[list-item]
    done = 0

    # ── Process each video — direct seek per target frame ────────────
    # Sampled frames are sparse (one per ~10s), so seeking is much faster
    # than decoding every intermediate frame sequentially.
    for vpath, frame_idx_map in video_frame_map.items():
        cap = cv2.VideoCapture(vpath)

        frame_batch: List[torch.Tensor]   = []
        frame_pils:  List[PILImage.Image] = []
        frame_fidxs: List[int]            = []

        def _flush_batch():
            nonlocal done
            if not frame_batch:
                return
            imgs = torch.stack(frame_batch).to(device)  # [B, C, H, W]
            with torch.no_grad():
                patches = vision_encoder.forward_features(
                    imgs, norm=True, strip_cls_token=True
                )  # [B, N, D]

            for b, (fidx_b, pil_b) in enumerate(zip(frame_fidxs, frame_pils)):
                pt = patches[b]  # [N, D]

                for sample_idx in frame_idx_map[fidx_b]:
                    _, _, cx, cy, w, h, fw, fh = samples[sample_idx]

                    # Convert center coords → top-left, apply context_scale
                    ew = w * context_scale
                    eh = h * context_scale
                    x1 = max(0, int(cx - ew / 2))
                    y1 = max(0, int(cy - eh / 2))
                    x2 = min(fw, int(cx + ew / 2))
                    y2 = min(fh, int(cy + eh / 2))
                    if x2 <= x1:
                        x2 = x1 + 1
                    if y2 <= y1:
                        y2 = y1 + 1

                    bbox = BBoxPrompt(
                        pixel_coords=(x1, y1, x2 - x1, y2 - y1),
                        image_size=(fw, fh),
                    )
                    crop = pil_b.crop((x1, y1, x2, y2))

                    with torch.no_grad():
                        teacher = pe_model.encode_image(
                            image_transform(crop).unsqueeze(0).to(device),
                            normalize=True,
                        )[0]  # [D]

                    patches_mm[sample_idx]  = pt.cpu().half().numpy()
                    teachers_mm[sample_idx] = teacher.cpu().half().numpy()
                    all_bboxes[sample_idx]  = bbox
                    done += 1

            frame_batch.clear()
            frame_pils.clear()
            frame_fidxs.clear()
            print(f"\r  {done}/{n_samples} samples", end="", flush=True)

        # Seek directly to each target frame (sorted to minimise seek distance)
        for fidx in sorted(frame_idx_map.keys()):
            cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
            ret, bgr = cap.read()
            if not ret:
                print(f"\n  Warning: could not read frame {fidx} from {vpath}")
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            pil = PILImage.fromarray(rgb)
            frame_batch.append(image_transform(pil))
            frame_pils.append(pil)
            frame_fidxs.append(fidx)
            if len(frame_batch) >= batch_size:
                _flush_batch()

        _flush_batch()
        cap.release()

    patches_mm.flush()
    teachers_mm.flush()
    print(f"\r  Precomputed {done}/{n_samples} samples → {tmp_dir}")

    # Re-open as read-only for zero-copy lazy page-in during training
    patches_ro  = np.memmap(tmp_dir / "patches.bin",  dtype="float16", mode="r",
                             shape=(n_samples, N_patches, D_patch))
    teachers_ro = np.memmap(tmp_dir / "teachers.bin", dtype="float16", mode="r",
                             shape=(n_samples, D_teacher))

    return CachedDataset(
        patch_tokens   = torch.from_numpy(patches_ro),
        teacher_embeds = torch.from_numpy(teachers_ro),
        bboxes         = all_bboxes,
    )


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(
    head:        PositionCrossAttention,
    dataset:     CachedDataset,
    patch_grid:  torch.Tensor,
    proj:        Optional[torch.Tensor],
    device:      torch.device,
    epochs:      int,
    lr:          float,
    batch_size:  int,
    num_workers: int,
    out_path:    str,
    use_amp:     bool = True,
) -> None:
    use_amp = use_amp and device.type == "cuda"

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size        = batch_size,
        shuffle           = True,
        num_workers       = num_workers,
        pin_memory        = True,
        persistent_workers= (num_workers > 0),
        collate_fn        = _collate_cached,
    )

    optimizer = torch.optim.AdamW(head.parameters(), lr=lr)
    scaler    = torch.cuda.amp.GradScaler(enabled=use_amp)

    print(f"\nTraining {len(dataset)} samples  "
          f"batch={batch_size}  epochs={epochs}  lr={lr}")
    print("─" * 50)

    for epoch in range(epochs):
        head.train()
        total_loss = 0.0

        for patch_tokens, bboxes, teachers in loader:
            patch_tokens = patch_tokens.to(device, non_blocking=True)  # [B, N, D]
            teachers     = teachers.to(device, non_blocking=True)      # [B, D]

            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=use_amp):
                student = head_forward(head, patch_tokens, bboxes, patch_grid, proj, device)
                loss    = (1.0 - F.cosine_similarity(student, teachers)).mean()

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()

        print(f"Epoch {epoch + 1:3d}/{epochs}  "
              f"loss={total_loss / len(loader):.6f}")

    torch.save(head.state_dict(), out_path)
    print(f"\nSaved → {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(
        description="Train PositionCrossAttention head via distillation from video tracks.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--data-dir",  required=True, metavar="DIR",
                   help="Folder with paired video + JSON track files "
                        "(1.MP4 + 1.json, 2.MP4 + 2.json, …).")
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
    p.add_argument("--num-heads", type=int, default=8)
    p.add_argument("--epochs",    type=int, default=10)
    p.add_argument("--lr",        type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--sample-interval", type=float, default=10.0,
                   help="Sample one frame per identity every N seconds (default: 10.0).")
    p.add_argument("--context-scale", type=float, default=1.0,
                   help="BBox expansion factor, must match inference scripts (default: 1.0).")
    p.add_argument("--precompute-batch-size", type=int, default=16,
                   help="Frames per GPU batch during precomputation (default: 16).")
    p.add_argument("--no-amp",    action="store_true",
                   help="Disable bf16 autocast.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args   = _parse_args()
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from core.vision_encoder.pe import CLIP
    from core.vision_encoder.transforms import get_image_transform

    # ── Load PE model ─────────────────────────────────────────────────
    print(f"Loading {args.model} (pretrained={not args.no_pretrained}) …")
    pe_model = CLIP.from_config(
        args.model,
        pretrained     = not args.no_pretrained,
        checkpoint_path= args.checkpoint,
    ).to(device).eval()

    vision_encoder = pe_model.visual
    for p in vision_encoder.parameters():
        p.requires_grad_(False)

    image_size = vision_encoder.image_size
    patch_size = vision_encoder.patch_size
    width      = vision_encoder.width
    print(f"  image_size={image_size}  patch_size={patch_size}  width={width}")

    patch_grid = build_patch_grid(image_size, patch_size).to(device)
    proj: Optional[torch.Tensor] = None
    if hasattr(vision_encoder, "proj") and vision_encoder.proj is not None:
        proj = vision_encoder.proj.to(device)

    image_transform = get_image_transform(image_size)

    # ── Build cross-attention head ────────────────────────────────────
    head = PositionCrossAttention(embed_dim=width, num_heads=args.num_heads).to(device)
    if args.resume is not None:
        state = torch.load(args.resume, map_location=device, weights_only=True)
        missing, unexpected = head.load_state_dict(state, strict=False)
        if missing:
            print(f"  Warning — missing keys (random init): {missing}")
        if unexpected:
            print(f"  Warning — unexpected keys (ignored): {unexpected}")
        print(f"  Resumed from {args.resume}")

    # ── Scan data folder ──────────────────────────────────────────────
    samples = scan_video_folder(args.data_dir, sample_interval=args.sample_interval)

    # ── Precompute features ───────────────────────────────────────────
    dataset = precompute_from_videos(
        samples         = samples,
        vision_encoder  = vision_encoder,
        pe_model        = pe_model,
        image_transform = image_transform,
        device          = device,
        context_scale   = args.context_scale,
        batch_size      = args.precompute_batch_size,
    )

    # ── Train ─────────────────────────────────────────────────────────
    train(
        head        = head,
        dataset     = dataset,
        patch_grid  = patch_grid,
        proj        = proj,
        device      = device,
        epochs      = args.epochs,
        lr          = args.lr,
        batch_size  = args.batch_size,
        num_workers = args.num_workers,
        out_path    = args.out,
        use_amp     = not args.no_amp,
    )
