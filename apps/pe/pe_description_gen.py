"""
PE Description Generator
========================
For each tracked identity, generates a structured description covering:
  • Motion       — what the person's body is doing
  • Social       — how the person relates to others in the scene
  • Activity     — the high-level activity being performed

Descriptions are produced by comparing mean-pooled crop embeddings (from the
baseline PE vision encoder) against category-specific text candidates, then
selecting the best-matching label in each category.

Output: a single JSON file structured as:

    {
      "<identity_id>": [
        {
          "start_frame":         <int>,
          "end_frame":           <int>,
          "start_sec":           <float>,
          "end_sec":             <float>,
          "motion":              {"label": "...", "score": <float>},
          "social_interaction":  {"label": "...", "score": <float>},
          "activity":            {"label": "...", "score": <float>},
          "description":         "..."       ← short human-readable sentence
        },
        ...
      ],
      ...
    }

The "description" field is a composed sentence, e.g.:
  "A child walking alone and playing with objects."

Usage
-----
    python apps/pe/pe_description_gen.py \\
        --video       input.mp4 \\
        --track-file  tracks.json \\
        --image-size  1920 1080 \\
        --out         descriptions.json

    # With a custom window size (seconds):
    python apps/pe/pe_description_gen.py \\
        --video       input.mp4 \\
        --track-file  tracks.json \\
        --image-size  1920 1080 \\
        --window-sec  2.0 \\
        --out         descriptions.json

    # Expand the crop region around the bbox:
    python apps/pe/pe_description_gen.py \\
        --video         input.mp4 \\
        --track-file    tracks.json \\
        --image-size    1920 1080 \\
        --context-scale 1.5 \\
        --out           descriptions.json
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image as PILImage


# ---------------------------------------------------------------------------
# Category-specific text candidates
# ---------------------------------------------------------------------------

MOTION_LABELS: List[str] = [
    "a person standing still",
    "a person walking",
    "a person running",
    "a person sitting",
    "a person crouching or squatting",
    "a person jumping",
    "a person lying down",
    "a person crawling",
    "a person dancing or moving rhythmically",
    "a person making hand gestures",
]

SOCIAL_LABELS: List[str] = [
    "a person alone with no one nearby",
    "a person talking face to face with one other person",
    "a person interacting with a small group of peers",
    "a person in a large group",
    "a person interacting with an adult or teacher",
    "a person sitting or standing side by side with others",
    "a person watching others from a distance",
]

ACTIVITY_LABELS: List[str] = [
    "a person reading a book or document",
    "a person playing with toys or objects",
    "a person drawing writing or doing craftwork",
    "a person eating or drinking",
    "a person building or assembling something",
    "a person using a phone computer or electronic device",
    "a person exercising or doing physical activity",
    "a person talking or having a conversation",
    "a person exploring or looking around",
    "a person resting or doing nothing",
    "a person playing a musical instrument",
    "a person cleaning or tidying up",
]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate per-identity structured descriptions from a tracked video.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--video", required=True, metavar="PATH",
                   help="Input video file.")
    p.add_argument("--track-file", required=True, metavar="FILE",
                   help="Identity-format JSON: {id: [[frame,cx,cy,w,h], ...], ...}")
    p.add_argument("--image-size", required=True, type=int, nargs=2, metavar=("W", "H"),
                   help="Original frame size in pixels (width height).")
    p.add_argument("--model", default="PE-Core-G14-448", metavar="NAME",
                   help="PE model name (default: PE-Core-G14-448).")
    p.add_argument("--checkpoint", default=None, metavar="PATH",
                   help="PE checkpoint path (optional).")
    p.add_argument("--no-pretrained", action="store_true",
                   help="Skip loading pretrained weights (smoke test only).")
    p.add_argument("--window-sec", type=float, default=2.0, metavar="S",
                   help="Temporal window for description aggregation in seconds "
                        "(default: 2.0).")
    p.add_argument("--fps", type=float, default=None, metavar="N",
                   help="Override video frame rate (inferred from file when omitted).")
    p.add_argument("--context-scale", type=float, default=1.5, metavar="S",
                   help="Expand bbox by this factor before cropping (default: 1.5).")
    p.add_argument("--batch-size", type=int, default=16, metavar="N",
                   help="Number of crops to encode per batch (default: 16).")
    p.add_argument("--max-frames", type=int, default=None, metavar="N",
                   help="Stop after this many frames (default: all frames).")
    p.add_argument("--softmax", action="store_true",
                   help="Use softmax-normalised scores instead of raw cosine "
                        "similarities when selecting the best label.")
    p.add_argument("--out", default="descriptions.json", metavar="PATH",
                   help="Output JSON file path (default: descriptions.json).")
    p.add_argument("--pretty", action="store_true",
                   help="Pretty-print the output JSON (indent=2).")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Track loading
# ---------------------------------------------------------------------------

def _load_tracks(path: str) -> Dict[str, List]:
    """Return identity → list of [frame_idx, cx, cy, w, h] sorted by frame."""
    with open(path) as f:
        data = json.load(f)
    return {
        identity: sorted(entries, key=lambda e: e[0])
        for identity, entries in data.items()
    }


def _build_frame_map(tracks: Dict[str, List]) -> Dict[int, List[Tuple]]:
    """Return frame_idx → list of (identity, cx, cy, w, h)."""
    frame_map: Dict[int, List] = defaultdict(list)
    for identity, entries in tracks.items():
        for entry in entries:
            fidx, cx, cy, w, h = entry
            frame_map[int(fidx)].append((identity, cx, cy, w, h))
    return dict(frame_map)


# ---------------------------------------------------------------------------
# Crop helper
# ---------------------------------------------------------------------------

def _crop_bbox(
    pil_image: PILImage.Image,
    cx: float, cy: float, w: float, h: float,
    context_scale: float,
    frame_w: int, frame_h: int,
) -> PILImage.Image:
    ew = w * context_scale
    eh = h * context_scale
    x1 = max(0, int(cx - ew / 2))
    y1 = max(0, int(cy - eh / 2))
    x2 = min(frame_w, int(cx + ew / 2))
    y2 = min(frame_h, int(cy + eh / 2))
    if x2 <= x1 or y2 <= y1:
        x1 = max(0, int(cx) - 1)
        y1 = max(0, int(cy) - 1)
        x2 = min(frame_w, x1 + 2)
        y2 = min(frame_h, y1 + 2)
    return pil_image.crop((x1, y1, x2, y2))


# ---------------------------------------------------------------------------
# Text encoding
# ---------------------------------------------------------------------------

def _encode_labels(
    pe_model: "CLIP",  # type: ignore[name-defined]
    tokenizer,
    labels: List[str],
    device: torch.device,
) -> torch.Tensor:
    """Encode a list of text labels; return normalised [N, E] tensor."""
    with torch.no_grad():
        return pe_model.encode_text(tokenizer(labels).to(device), normalize=True)


# ---------------------------------------------------------------------------
# Best-match selection
# ---------------------------------------------------------------------------

def _best_match(
    feat: torch.Tensor,         # [E] normalised
    text_feats: torch.Tensor,   # [N, E] normalised
    labels: List[str],
    use_softmax: bool,
    device: torch.device,
) -> Tuple[str, float]:
    """Return (best_label, score) for *feat* against *text_feats*."""
    cos = (feat.to(device) @ text_feats.T).cpu()   # [N]
    display = F.softmax(cos, dim=0) if use_softmax else cos
    best_idx = int(display.argmax())
    return labels[best_idx], float(display[best_idx])


# ---------------------------------------------------------------------------
# Description composer
# ---------------------------------------------------------------------------

def _compose_description(motion: str, social: str, activity: str) -> str:
    """
    Build a short human-readable sentence from the three category winners.

    The raw labels start with "a person …"; we strip the prefix and assemble:
        "A person <motion>, <social>, and <activity>."
    """
    def _strip(label: str) -> str:
        for prefix in ("a person ", "a child "):
            if label.lower().startswith(prefix):
                return label[len(prefix):]
        return label

    m = _strip(motion)
    s = _strip(social)
    a = _strip(activity)
    return f"A person {m}, {s}, and {a}."


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = _parse_args()

    try:
        import cv2
    except ImportError:
        sys.exit("opencv-python is required.  Install: pip install opencv-python-headless")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frame_w, frame_h = args.image_size

    # ------------------------------------------------------------------
    # 1. Load PE model
    # ------------------------------------------------------------------
    print(f"Loading PE model ({args.model}) …")
    from core.vision_encoder.pe import CLIP
    from core.vision_encoder.transforms import get_image_transform, get_text_tokenizer

    pretrained = not args.no_pretrained
    pe_model = CLIP.from_config(
        args.model, pretrained=pretrained, checkpoint_path=args.checkpoint
    ).to(device).eval()

    enc_image_size: int = pe_model.visual.image_size
    img_transform = get_image_transform(enc_image_size)
    tokenizer = get_text_tokenizer(pe_model.context_length)
    print(f"  encoder image size: {enc_image_size}px")

    # ------------------------------------------------------------------
    # 2. Encode category text candidates
    # ------------------------------------------------------------------
    print("Encoding category labels …")
    motion_feats = _encode_labels(pe_model, tokenizer, MOTION_LABELS, device)
    social_feats = _encode_labels(pe_model, tokenizer, SOCIAL_LABELS, device)
    activity_feats = _encode_labels(pe_model, tokenizer, ACTIVITY_LABELS, device)
    print(f"  motion: {len(MOTION_LABELS)} labels")
    print(f"  social: {len(SOCIAL_LABELS)} labels")
    print(f"  activity: {len(ACTIVITY_LABELS)} labels")

    # ------------------------------------------------------------------
    # 3. Load tracks
    # ------------------------------------------------------------------
    print(f"Loading tracks from {args.track_file} …")
    tracks = _load_tracks(args.track_file)
    frame_map = _build_frame_map(tracks)
    print(f"  {len(tracks)} identities, "
          f"{sum(len(v) for v in tracks.values())} total track entries")

    # ------------------------------------------------------------------
    # 4. Probe video
    # ------------------------------------------------------------------
    cap_probe = cv2.VideoCapture(args.video)
    if not cap_probe.isOpened():
        sys.exit(f"Cannot open video: {args.video}")
    vid_fps_probe = cap_probe.get(cv2.CAP_PROP_FPS) or 30.0
    vid_total = int(cap_probe.get(cv2.CAP_PROP_FRAME_COUNT))
    cap_probe.release()

    fps: float = args.fps or vid_fps_probe
    window_frames = max(1, int(args.window_sec * fps))
    max_frames = args.max_frames or vid_total
    print(f"  fps={fps:.3f}  window={args.window_sec}s ({window_frames} frames)  "
          f"max_frames={max_frames}")

    # ------------------------------------------------------------------
    # 5. Read video — encode crops per identity per frame
    # ------------------------------------------------------------------
    # identity → {frame_idx: feature [E] (cpu)}
    identity_frame_feats: Dict[str, Dict[int, torch.Tensor]] = {
        ident: {} for ident in tracks
    }

    print(f"\nEncoding crops from {args.video} …")
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        sys.exit(f"Cannot open video: {args.video}")

    batch_crops: List[torch.Tensor] = []
    batch_keys: List[Tuple[str, int]] = []

    def _flush() -> None:
        if not batch_crops:
            return
        imgs = torch.stack(batch_crops).to(device)
        with torch.no_grad():
            feats = pe_model.encode_image(imgs, normalize=True)   # [B, E]
        for (ident, fidx), feat in zip(batch_keys, feats):
            identity_frame_feats[ident][fidx] = feat.cpu()
        batch_crops.clear()
        batch_keys.clear()

    frame_idx = 0
    while frame_idx < max_frames:
        ret, bgr = cap.read()
        if not ret:
            break

        if frame_idx in frame_map:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            pil_frame = PILImage.fromarray(rgb)

            for (ident, cx, cy, w, h) in frame_map[frame_idx]:
                crop = _crop_bbox(pil_frame, cx, cy, w, h,
                                  args.context_scale, frame_w, frame_h)
                batch_crops.append(img_transform(crop))
                batch_keys.append((ident, frame_idx))
                if len(batch_crops) >= args.batch_size:
                    _flush()

        frame_idx += 1
        if frame_idx % 30 == 0 or frame_idx == max_frames:
            print(f"\r  {frame_idx}/{max_frames} frames", end="", flush=True)

    _flush()
    cap.release()
    print()

    # ------------------------------------------------------------------
    # 6. Per-identity, per-window: compute descriptions
    # ------------------------------------------------------------------
    results: Dict[str, List[Dict]] = {}

    for ident, entries in tracks.items():
        feat_map = identity_frame_feats[ident]
        if not feat_map:
            print(f"  [{ident}] no encoded frames — skipping")
            continue

        # Group frames into window buckets
        buckets: Dict[int, List[torch.Tensor]] = defaultdict(list)
        for entry in entries:
            fidx = int(entry[0])
            if fidx >= max_frames:
                continue
            wid = fidx // window_frames
            if fidx in feat_map:
                buckets[wid].append(feat_map[fidx])

        windows: List[Dict] = []
        for wid in sorted(buckets):
            feats_list = buckets[wid]
            if not feats_list:
                continue

            # Mean-pool over window frames → normalised aggregate
            mean_feat = F.normalize(
                torch.stack(feats_list).mean(dim=0), dim=-1
            )  # [E]

            motion_label, motion_score = _best_match(
                mean_feat, motion_feats, MOTION_LABELS, args.softmax, device
            )
            social_label, social_score = _best_match(
                mean_feat, social_feats, SOCIAL_LABELS, args.softmax, device
            )
            activity_label, activity_score = _best_match(
                mean_feat, activity_feats, ACTIVITY_LABELS, args.softmax, device
            )

            description = _compose_description(motion_label, social_label, activity_label)

            start_fr = wid * window_frames
            end_fr   = (wid + 1) * window_frames - 1
            windows.append({
                "start_frame":        start_fr,
                "end_frame":          end_fr,
                "start_sec":          round(start_fr / fps, 3),
                "end_sec":            round((end_fr + 1) / fps, 3),
                "n_frames":           len(feats_list),
                "motion": {
                    "label": motion_label,
                    "score": round(motion_score, 4),
                },
                "social_interaction": {
                    "label": social_label,
                    "score": round(social_score, 4),
                },
                "activity": {
                    "label": activity_label,
                    "score": round(activity_score, 4),
                },
                "description": description,
            })

        results[ident] = windows

        # Console summary
        print(f"\n[{ident}]")
        for w in windows:
            print(f"  {w['start_sec']:6.1f}s – {w['end_sec']:6.1f}s "
                  f"({w['n_frames']:3d} frames)")
            print(f"    motion:   {w['motion']['label']}  ({w['motion']['score']:.3f})")
            print(f"    social:   {w['social_interaction']['label']}"
                  f"  ({w['social_interaction']['score']:.3f})")
            print(f"    activity: {w['activity']['label']}"
                  f"  ({w['activity']['score']:.3f})")
            print(f"    → {w['description']}")

    # ------------------------------------------------------------------
    # 7. Write JSON
    # ------------------------------------------------------------------
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    indent = 2 if args.pretty else None
    with open(out_path, "w") as f:
        json.dump(results, f, indent=indent, ensure_ascii=False)

    print(f"\nSaved → {out_path}  ({len(results)} identities)")
