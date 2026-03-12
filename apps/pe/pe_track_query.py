"""
PE Track Query  —  fast per-track feature extraction from pre-saved patch tokens.

The PE visual encoder is NEVER loaded here.  Only two lightweight components
are needed:

    1. PositionCrossAttention head  (~few MB, from a .pt checkpoint)
    2. PE text encoder              (loaded only when --text is supplied)

This means you can query thousands of tracks against arbitrary text in seconds,
after a one-time offline run of pe_extract_patch_features.py.

Pipeline
--------
    patch_features.pt  ──►  patch_tokens[t]  ──►  cross-attn head
                                                        │
         track.json     ──►  BBoxPrompt[t]  ──►  query patches
                                                        │
                                               per-frame feat [D]
                                                        │
                                              mean-pool + L2-norm
                                                        │
                                              clip embed [D]
                                                        │
             --text "..."  ──►  text encoder  ──►  similarity score

Track file formats
-------------------
  Identity JSON  (tracker output — select identity with --identity):
    {
      "d14717": [[frame, x, y, w, h], [frame, x, y, w, h], ...],
      "d14718": [...],
      ...
    }
    Requires --image-size W H and --identity <id>.

  Compact JSON  (paired with feature file's sorted frame list):
    {"image_size": [W, H], "track": [[x,y,w,h], [x,y,w,h], ...]}

  Explicit JSON  (filenames cross-referenced against feature file):
    [{"file": "frame_000.jpg", "bbox": [x,y,w,h], "image_size": [W,H]}, ...]

  TXT  (requires --image-size W H):
    frame_000.jpg  x  y  w  h

Usage
-----
    # Basic: features + track → per-frame embeddings
    python apps/pe/pe_track_query.py \\
        --features     patch_features.pt \\
        --track-file   track.json \\
        --head-checkpoint head.pt

    # Identity-format track (tracker output):
    python apps/pe/pe_track_query.py \\
        --features     patch_features.pt \\
        --track-file   tracks.json \\
        --identity     d14717 \\
        --image-size   1920 1080 \\
        --head-checkpoint head.pt \\
        --text "a person walking"

    # With text comparison:
    python apps/pe/pe_track_query.py \\
        --features     patch_features.pt \\
        --track-file   track.json \\
        --head-checkpoint head.pt \\
        --text "a person walking" "a dog running"

    # Save per-frame + mean features for downstream use:
    python apps/pe/pe_track_query.py \\
        --features     patch_features.pt \\
        --track-file   track.json \\
        --head-checkpoint head.pt \\
        --out          track_feats.pt

    # Output file contains:
    #   per_frame : Tensor[T, D]  — L2-normalized per-frame embeddings
    #   mean_feat : Tensor[D]     — L2-normalized clip embedding (mean-pooled)
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn.functional as F


def _parse_args():
    p = argparse.ArgumentParser(
        description="Fast track query on pre-saved PE patch features.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--features", required=True, metavar="PATH",
                   help="Pre-computed patch features (.pt) from pe_extract_patch_features.py.")
    p.add_argument("--track-file", required=True, metavar="FILE",
                   help="Track file: JSON or TXT mapping frames to bboxes.")
    p.add_argument("--head-checkpoint", default=None, metavar="PATH",
                   help="PositionCrossAttention weights (.pt). "
                        "Omit only for smoke tests (random head).")
    p.add_argument("--num-heads", type=int, default=8,
                   help="Cross-attention heads — must match training (default: 8).")
    p.add_argument("--text", nargs="+", default=None, metavar="PHRASE",
                   help="Text phrases to compare with the mean track embedding.")
    p.add_argument("--model", default=None, metavar="NAME",
                   help="PE model name for the text encoder "
                        "(inferred from features file when omitted).")
    p.add_argument("--checkpoint", default=None, metavar="PATH",
                   help="PE checkpoint path for the text encoder.")
    p.add_argument("--no-pretrained", action="store_true",
                   help="Skip loading pretrained weights (smoke test).")
    p.add_argument("--identity", default=None, metavar="ID",
                   help="Identity key to query from an identity-format track file "
                        "(e.g. 'd14717'). List available IDs by omitting this flag.")
    p.add_argument("--image-size", type=int, nargs=2, default=None, metavar=("W", "H"),
                   help="Original frame size in pixels — required for identity-format "
                        "and .txt track files.")
    p.add_argument("--frame-stride", type=int, default=1, metavar="N",
                   help="Process every Nth frame of the track (default: 1).")
    p.add_argument("--batch-size", type=int, default=32, metavar="N",
                   help="Frames per head-forward batch (default: 32).")
    p.add_argument("--out", default=None, metavar="PATH",
                   help="Save {per_frame [T,D], mean_feat [D]} as a .pt file.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Minimal head-only forward (mirrors PEBBoxFeatureExtractor._forward_from_patches)
# ---------------------------------------------------------------------------

def _head_forward_batch(
    head: "torch.nn.Module",
    patch_tokens: torch.Tensor,    # [B, N, D]
    bboxes: list,
    patch_grid: torch.Tensor,      # [N, 2]
    proj: Optional[torch.Tensor],  # [D, E] | None
    device: torch.device,
) -> torch.Tensor:                 # [B, E]  L2-normalized
    """Run the cross-attention head on a batch of pre-saved patch tokens."""
    from pe_position_approach1 import select_bbox_patches

    B, _, D = patch_tokens.shape
    tokens  = patch_tokens.to(device)

    query_list = []
    for b, bbox in enumerate(bboxes):
        idx = select_bbox_patches(bbox, patch_grid)
        query_list.append(tokens[b, idx])   # [k, D]

    max_k        = max(q.shape[0] for q in query_list)
    query_padded = torch.zeros(B, max_k, D, device=device)
    for b, q in enumerate(query_list):
        query_padded[b, : q.shape[0]] = q

    embeds = head(query_padded, tokens)   # [B, D]

    if proj is not None:
        embeds = embeds @ proj            # [B, E]

    return F.normalize(embeds, dim=-1)   # [B, E]


# ---------------------------------------------------------------------------
# Identity-format track loader
# ---------------------------------------------------------------------------

def _is_identity_format(data: object) -> bool:
    """Return True if data is a dict mapping identity strings to [[frame,x,y,w,h],…]."""
    if not isinstance(data, dict):
        return False
    # Heuristic: first value must be a non-empty list whose first element is
    # itself a list/tuple of 5 numbers starting with a frame index.
    for v in data.values():
        if not isinstance(v, list) or not v:
            return False
        first = v[0]
        return isinstance(first, (list, tuple)) and len(first) == 5
    return False


def _load_identity_track(
    path: str,
    identity: "str | None",
    image_size: "Tuple[int, int] | None",
) -> "tuple[list[str], list]":
    """
    Parse an identity-format track file.

    Format::

        {
          "d14717": [[frame, x, y, w, h], ...],
          "d14718": [[frame, x, y, w, h], ...],
        }

    Parameters
    ----------
    path        : path to the JSON file
    identity    : which ID to load; if None, lists available IDs and exits
    image_size  : (W, H) of the original frames — required for BBoxPrompt

    Returns
    -------
    frame_keys  : list of zero-padded frame-index strings (e.g. ["000000", "000003"])
    bboxes      : list of BBoxPrompt, one per frame
    """
    import json
    from pe_position_approach1 import BBoxPrompt

    with open(path) as f:
        data = json.load(f)

    if identity is None:
        ids = sorted(data.keys())
        print("Available identities in track file:")
        for id_ in ids:
            print(f"  {id_}  ({len(data[id_])} frames)")
        sys.exit("Re-run with --identity <id>")

    if identity not in data:
        sys.exit(f"Identity {identity!r} not found. "
                 f"Available: {sorted(data.keys())}")

    if image_size is None:
        sys.exit("--image-size W H is required for identity-format track files.")

    entries = data[identity]  # [[frame, x, y, w, h], ...]
    # Sort by frame index in case the tracker output is unordered
    entries = sorted(entries, key=lambda e: e[0])

    frame_keys: list[str] = []
    bboxes: list          = []
    for entry in entries:
        frame_idx, x, y, w, h = entry
        frame_keys.append(f"{int(frame_idx):06d}")
        bboxes.append(BBoxPrompt(
            pixel_coords=(int(x), int(y), int(w), int(h)),
            image_size=image_size,
        ))

    return frame_keys, bboxes


# ---------------------------------------------------------------------------
# Track alignment helpers
# ---------------------------------------------------------------------------

def _align_track_to_features(
    track_paths: list[str],
    feat_paths:  list[str],
) -> list[int]:
    """
    Return the feature-file index for each entry in the track.

    Supports both sources:
    - Frame-directory features : feat_paths are filenames, matched on stem.
    - Video features            : feat_paths are zero-padded frame indices
                                  (e.g. "000042"); track_paths matched the
                                  same way after stripping extension/path.
    """
    # Build lookup: bare key (no dir, no extension) → index
    key_to_idx = {Path(p).stem: i for i, p in enumerate(feat_paths)}
    # Also accept exact name match (handles "frame_000.jpg" → "frame_000.jpg")
    key_to_idx.update({Path(p).name: i for i, p in enumerate(feat_paths)})

    indices: list[int] = []
    for tp in track_paths:
        # Try stem first (strips extension), then full name, then as-is
        for candidate in (Path(tp).stem, Path(tp).name, str(tp)):
            if candidate in key_to_idx:
                indices.append(key_to_idx[candidate])
                break
        else:
            raise ValueError(
                f"Track frame {tp!r} not found in the feature file.\n"
                "Make sure pe_extract_patch_features.py was run on the same source."
            )
    return indices


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = _parse_args()

    # Make pe_position_approach1 importable from the same directory
    sys.path.insert(0, str(Path(__file__).parent))
    from pe_position_approach1 import (
        build_patch_grid,
        load_track_file,
        PositionCrossAttention,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------
    # 1. Load pre-saved patch tokens  (no image encoder, no image I/O)
    # ------------------------------------------------------------------
    print(f"Loading patch features from {args.features} …")
    feat_data = torch.load(args.features, map_location="cpu", weights_only=True)

    all_patch_tokens: torch.Tensor      = feat_data["patch_tokens"]   # [T_all, N, D]
    proj: Optional[torch.Tensor]        = feat_data.get("proj")       # [D, E] | None
    # Support both old key ("frame_paths") and current key ("frame_keys")
    feat_frame_paths: list[str] = feat_data.get("frame_keys") or feat_data["frame_paths"]
    model_name: str                     = feat_data["model_name"]
    image_size: int                     = feat_data["image_size"]
    patch_size: int                     = feat_data["patch_size"]
    width: int                          = feat_data["width"]

    T_all = all_patch_tokens.shape[0]
    print(f"  {T_all} frames  |  patch_tokens {tuple(all_patch_tokens.shape)}"
          f"  dtype={all_patch_tokens.dtype}")

    if proj is not None:
        proj = proj.to(device)

    # ------------------------------------------------------------------
    # 2. Load cross-attention head  (tiny — the only model component needed
    #    for feature extraction when text comparison is not requested)
    # ------------------------------------------------------------------
    head = PositionCrossAttention(embed_dim=width, num_heads=args.num_heads).to(device)

    if args.head_checkpoint is not None:
        state = torch.load(args.head_checkpoint, map_location=device, weights_only=True)
        head.load_state_dict(state)
        print(f"  Head weights loaded ← {args.head_checkpoint}")
    else:
        print("  Warning: --head-checkpoint not provided — using random head (smoke test only).")

    head.eval()

    patch_grid = build_patch_grid(image_size, patch_size).to(device)

    # ------------------------------------------------------------------
    # 3. Parse track  +  align with the feature file
    # ------------------------------------------------------------------
    img_size_arg: Optional[Tuple[int, int]] = (
        tuple(args.image_size) if args.image_size else None   # type: ignore[arg-type]
    )

    import json as _json
    with open(args.track_file) as _f:
        _raw = _json.load(_f)

    if _is_identity_format(_raw):
        track_paths, track_bboxes = _load_identity_track(
            args.track_file,
            identity=args.identity,
            image_size=img_size_arg,
        )
    else:
        track_paths, track_bboxes = load_track_file(
            args.track_file,
            image_size=img_size_arg,
        )

    # Cross-reference track frame keys → feature-file indices
    feat_indices = _align_track_to_features(track_paths, feat_frame_paths)

    # Apply stride
    stride       = max(1, args.frame_stride)
    feat_indices = feat_indices[::stride]
    track_bboxes = track_bboxes[::stride]
    T = len(track_bboxes)

    print(f"Track : {T} frames  (stride={stride},  "
          f"covering {feat_indices[0]}–{feat_indices[-1]} of {T_all} feature frames)")

    # ------------------------------------------------------------------
    # 4. Per-frame cross-attention  (head only — very fast)
    # ------------------------------------------------------------------
    print(f"Running cross-attention head (batch_size={args.batch_size}) …")

    per_frame_list: list[torch.Tensor] = []

    with torch.no_grad():
        for start in range(0, T, args.batch_size):
            end      = min(start + args.batch_size, T)
            b_tokens = all_patch_tokens[feat_indices[start:end]]   # [B, N, D]
            b_bboxes = track_bboxes[start:end]

            feats = _head_forward_batch(
                head, b_tokens, b_bboxes, patch_grid, proj, device
            )  # [B, E]
            per_frame_list.append(feats.cpu())

    per_frame = torch.cat(per_frame_list, dim=0)            # [T, E]
    mean_feat = F.normalize(per_frame.mean(dim=0), dim=-1)  # [E]

    E = per_frame.shape[1]
    print(f"\nper_frame : {per_frame.shape}  (L2 norms ≈ 1.0)")
    print(f"mean_feat : {mean_feat.shape}   norm={mean_feat.norm().item():.6f}")

    # Frame-to-frame cosine similarity  (motion smoothness proxy)
    if T > 1:
        consec = F.cosine_similarity(per_frame[:-1], per_frame[1:], dim=-1)
        print(f"frame-to-frame sim : mean={consec.mean():.4f}  "
              f"min={consec.min():.4f}  max={consec.max():.4f}")

    # ------------------------------------------------------------------
    # 5. Text comparison  (loads PE text encoder — visual encoder skipped)
    # ------------------------------------------------------------------
    if args.text is not None:
        mname      = args.model or model_name
        pretrained = not args.no_pretrained

        print(f"\nLoading PE text encoder ({mname}) …")
        from core.vision_encoder.pe import CLIP
        from core.vision_encoder.transforms import get_text_tokenizer

        pe_model = CLIP.from_config(
            mname, pretrained=pretrained, checkpoint_path=args.checkpoint
        ).to(device).eval()

        # Drop the visual encoder — we only need text here
        del pe_model.visual
        torch.cuda.empty_cache()

        tokenizer  = get_text_tokenizer(pe_model.context_length)
        tokens     = tokenizer(args.text).to(device)
        with torch.no_grad():
            text_feats = pe_model.encode_text(tokens, normalize=True)  # [Q, E]

        logit_scale = pe_model.logit_scale.exp().item()

        # Mean track embedding vs. text
        mean_sims = mean_feat.to(device) @ text_feats.T          # [Q]

        # Per-frame vs. text
        frame_sims = per_frame.to(device) @ text_feats.T         # [T, Q]

        col_w = max(len(t) for t in args.text) + 2
        print("\n--- Mean track embedding ↔ text ---")
        print(f"  {'Phrase':<{col_w}}  {'Similarity':>10}  {'Logit':>11}"
              f"  {'Frame mean':>11}  {'Frame min':>10}  {'Frame max':>10}")
        print(f"  {'-'*col_w}  {'----------':>10}  {'-----------':>11}"
              f"  {'-----------':>11}  {'----------':>10}  {'----------':>10}")
        for q, phrase in enumerate(args.text):
            ms = mean_sims[q].item()
            fs = frame_sims[:, q]
            print(
                f"  {phrase:<{col_w}}"
                f"  {ms:>+10.4f}"
                f"  {ms * logit_scale:>+11.4f}"
                f"  {fs.mean().item():>+11.4f}"
                f"  {fs.min().item():>+10.4f}"
                f"  {fs.max().item():>+10.4f}"
            )

    # ------------------------------------------------------------------
    # 6. Save results
    # ------------------------------------------------------------------
    if args.out is not None:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"per_frame": per_frame, "mean_feat": mean_feat},
            out_path,
        )
        print(f"\nResults saved → {out_path}")
        print(f"  per_frame : {per_frame.shape}")
        print(f"  mean_feat : {mean_feat.shape}")
