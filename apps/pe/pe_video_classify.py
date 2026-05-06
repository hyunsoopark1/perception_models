"""
PE-Core Zero-Shot Video Classifier
====================================
Classifies Motion / Social / Activity for each tracked identity using
PE-Core (a CLIP-style vision-language model) — no LLM required.

For each window, for each tracked identity:
  1. Crop the bbox region from each frame (hard pixel crop + context expansion).
  2. Encode each crop with PE-Core's vision encoder.
  3. Temporal mean-pool frame embeddings into one window-level embedding.
  4. Zero-shot classify via cosine similarity against pre-computed text label embeddings.

Labels are the same taxonomy as pe_description_gen.py (body_state, obj_verb, obj_noun,
social, safety_event) so outputs can be directly compared.

Output JSON
-----------
    {
      "<identity_id>": [
        {
          "start_frame":  <int>,
          "end_frame":    <int>,
          "start_sec":    <float>,
          "end_sec":      <float>,
          "n_frames":     <int>,
          "body_state":   "<label>",
          "obj_verb":     "<label>",
          "obj_noun":     "<label>",
          "social":       "<label>",
          "safety_event": "<label>",
          "top_k": {
            "body_state":   [["<label>", <score>], ...],
            "obj_verb":     [...],
            "social":       [...],
            "safety_event": [...]
          }
        },
        ...
      ],
      ...
    }

Usage
-----
    python apps/pe/pe_video_classify.py \\
        --video       input.mp4 \\
        --track-file  tracks.json \\
        --out         classifications.json \\
        --out-video   classify_overlay.mp4

    # Larger model, no video:
    python apps/pe/pe_video_classify.py \\
        --video       input.mp4 \\
        --track-file  tracks.json \\
        --model       PE-Core-G14-448 \\
        --no-video \\
        --out         classifications.json

    # Show top-3 labels per slot in video overlay:
    python apps/pe/pe_video_classify.py \\
        --video       input.mp4 \\
        --track-file  tracks.json \\
        --top-k       3
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image as PILImage


# ---------------------------------------------------------------------------
# Taxonomy label sets  (same as pe_description_gen.py)
# ---------------------------------------------------------------------------

BODY_STATES = [
    "idle_stand", "idle_sit", "walk", "walk_loaded", "run",
    "bend", "squat", "kneel", "reach_overhead", "reach_low",
    "twist", "crouch_sustained", "climb", "fall", "recover_balance",
]
OBJ_VERBS = [
    "reach", "grasp", "lift", "lower", "carry", "place", "push",
    "pull", "drag", "stack", "unstack", "pack", "unpack", "scan",
    "inspect", "operate", "throw", "catch", "rotate", "none",
]
OBJ_NOUNS = [
    "box", "pallet", "scanner", "cart", "forklift", "ladder",
    "tool", "document", "shelf", "bin", "bag", "package",
    "button", "screen", "door", "handle", "none",
]
SOCIAL_TAX = [
    "none", "talk", "handover", "receive", "co_manipulate",
    "gesture_instruct", "point",
]
SAFETY_EVENTS = [
    "none", "zone_enter", "zone_exit", "ppe_don", "ppe_doff",
    "near_miss", "hazard_response", "fall",
]

# Natural-language expansions for each label (used as CLIP text prompts)
_BODY_STATE_PHRASES = {
    "idle_stand":       "a person standing still",
    "idle_sit":         "a person sitting still",
    "walk":             "a person walking",
    "walk_loaded":      "a person walking while carrying a heavy load",
    "run":              "a person running",
    "bend":             "a person bending over at the waist",
    "squat":            "a person squatting down",
    "kneel":            "a person kneeling on the ground",
    "reach_overhead":   "a person reaching up overhead",
    "reach_low":        "a person reaching down to the ground",
    "twist":            "a person twisting their torso",
    "crouch_sustained": "a person crouching and holding the position",
    "climb":            "a person climbing a ladder or steps",
    "fall":             "a person falling or losing balance",
    "recover_balance":  "a person stumbling and recovering their balance",
}
_OBJ_VERB_PHRASES = {
    "reach":   "a person reaching toward an object",
    "grasp":   "a person grasping and gripping an object",
    "lift":    "a person lifting an object off the ground",
    "lower":   "a person lowering an object carefully",
    "carry":   "a person carrying an object",
    "place":   "a person placing an object down on a surface",
    "push":    "a person pushing an object",
    "pull":    "a person pulling an object",
    "drag":    "a person dragging an object along the floor",
    "stack":   "a person stacking boxes or objects",
    "unstack": "a person unstacking boxes or objects",
    "pack":    "a person packing items into a container",
    "unpack":  "a person unpacking items from a container",
    "scan":    "a person scanning a barcode on an item",
    "inspect": "a person visually inspecting an object",
    "operate": "a person operating a machine or device",
    "throw":   "a person throwing an object",
    "catch":   "a person catching a thrown object",
    "rotate":  "a person rotating or turning an object",
    "none":    "a person not touching or handling any object",
}
_SOCIAL_PHRASES = {
    "none":             "a person working alone with no social interaction",
    "talk":             "two people talking to each other",
    "handover":         "a person handing an object to another person",
    "receive":          "a person receiving an object from another person",
    "co_manipulate":    "two people jointly handling an object together",
    "gesture_instruct": "a person gesturing to instruct another person",
    "point":            "a person pointing at something to direct attention",
}
_OBJ_NOUN_PHRASES = {
    "box":       "a person handling a cardboard box",
    "pallet":    "a person handling a wooden pallet",
    "scanner":   "a person using a barcode scanner",
    "cart":      "a person pushing or pulling a cart",
    "forklift":  "a person operating a forklift",
    "ladder":    "a person using a ladder",
    "tool":      "a person using a hand tool",
    "document":  "a person handling papers or documents",
    "shelf":     "a person interacting with a storage shelf",
    "bin":       "a person handling a storage bin or tote",
    "bag":       "a person handling a bag",
    "package":   "a person handling a wrapped package",
    "button":    "a person pressing a button or switch",
    "screen":    "a person looking at or touching a screen",
    "door":      "a person opening or closing a door",
    "handle":    "a person gripping a handle or bar",
    "none":      "a person not interacting with any specific object",
}
_SAFETY_PHRASES = {
    "none":             "a normal safe working situation",
    "zone_enter":       "a person entering a restricted or hazardous zone",
    "zone_exit":        "a person exiting a restricted or hazardous zone",
    "ppe_don":          "a person putting on personal protective equipment",
    "ppe_doff":         "a person removing personal protective equipment",
    "near_miss":        "a near-miss safety incident almost occurring",
    "hazard_response":  "a person responding to a workplace hazard",
    "fall":             "a person falling and hitting the ground",
}

_SLOT_CONFIG = {
    "body_state":   (BODY_STATES,   _BODY_STATE_PHRASES),
    "obj_verb":     (OBJ_VERBS,     _OBJ_VERB_PHRASES),
    "obj_noun":     (OBJ_NOUNS,     _OBJ_NOUN_PHRASES),
    "social":       (SOCIAL_TAX,    _SOCIAL_PHRASES),
    "safety_event": (SAFETY_EVENTS, _SAFETY_PHRASES),
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="PE-Core zero-shot M/S/A classifier per identity per window.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--video",       required=True, metavar="PATH",
                   help="Input video file.")
    p.add_argument("--track-file",  required=True, metavar="FILE",
                   help="Identity-format JSON: {id: [[frame,cx,cy,w,h], ...], ...}")
    p.add_argument("--image-size",  default=None, type=int, nargs=2, metavar=("W", "H"),
                   help="Frame size in pixels (W H). Inferred from video when omitted.")
    p.add_argument("--model",       default="PE-Core-G14-448", metavar="NAME",
                   help="PE-Core model variant (default: PE-Core-G14-448).")
    p.add_argument("--checkpoint",  default=None, metavar="PATH",
                   help="Custom PE checkpoint path (overrides pretrained weights).")
    p.add_argument("--no-pretrained", action="store_true",
                   help="Skip loading pretrained weights (random init, for testing).")
    p.add_argument("--window-sec",  type=float, default=6.0, metavar="S",
                   help="Window duration in seconds (default: 6.0).")
    p.add_argument("--fps",         type=float, default=None, metavar="N",
                   help="Override video FPS (inferred from file when omitted).")
    p.add_argument("--max-frames",  type=int,   default=None, metavar="N",
                   help="Stop after this many frames (default: all).")
    p.add_argument("--max-minutes", type=float, default=None, metavar="M",
                   help="Stop after this many minutes (default: all).")
    p.add_argument("--num-frames",  type=int, default=8, metavar="N",
                   help="Frames sampled per window per identity for encoding (default: 8).")
    p.add_argument("--context-scale", type=float, default=1.5, metavar="S",
                   help="BBox expansion factor for crop (default: 1.5).")
    p.add_argument("--top-k",       type=int, default=3, metavar="K",
                   help="Number of top labels to store per slot (default: 3).")
    p.add_argument("--out",         default="classifications.json", metavar="PATH",
                   help="Output JSON path (default: classifications.json).")
    p.add_argument("--out-video",   default="classify_overlay.mp4", metavar="PATH",
                   help="Output annotated video (default: classify_overlay.mp4).")
    p.add_argument("--no-video",    action="store_true",
                   help="Skip video rendering.")
    p.add_argument("--font-scale",  type=float, default=0.42,
                   help="cv2 font scale for overlay (default: 0.42).")
    p.add_argument("--debug",       action="store_true",
                   help="Debug mode: process first 1 minute only and pretty-print JSON.")
    p.add_argument("--pretty",      action="store_true",
                   help="Pretty-print JSON output.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Track helpers
# ---------------------------------------------------------------------------

def _load_tracks(path: str) -> Dict[str, List]:
    with open(path) as f:
        data = json.load(f)
    return {
        ident: sorted(entries, key=lambda e: e[0])
        for ident, entries in data.items()
    }


def _build_frame_map(tracks: Dict[str, List]) -> Dict[int, List[Tuple]]:
    """frame_idx → [(identity, cx, cy, w, h)]"""
    fm: Dict[int, List] = defaultdict(list)
    for ident, entries in tracks.items():
        for entry in entries:
            fidx, cx, cy, w, h = entry
            fm[int(fidx)].append((ident, cx, cy, w, h))
    return dict(fm)


def _build_frame_annotations(tracks: Dict[str, List],
                              max_frames: int) -> Dict[int, List[Tuple]]:
    """frame_idx → [(identity, x1, y1, x2, y2)] corner format."""
    fa: Dict[int, List] = defaultdict(list)
    for ident, entries in tracks.items():
        for entry in entries:
            fidx, cx, cy, w, h = int(entry[0]), *entry[1:]
            if fidx >= max_frames:
                continue
            fa[fidx].append((ident,
                             int(cx - w / 2), int(cy - h / 2),
                             int(cx + w / 2), int(cy + h / 2)))
    return dict(fa)


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
# Text embedding pre-computation
# ---------------------------------------------------------------------------

def _build_label_embeddings(
    pe_model,
    tokenizer,
    device: torch.device,
) -> Dict[str, Tuple[List[str], torch.Tensor]]:
    """
    Pre-compute normalised text embeddings for all taxonomy slots.

    Returns:
        slot → (label_list, embeddings)  where embeddings is [N_labels, D]
    """
    slot_embeddings = {}
    with torch.no_grad():
        for slot, (labels, phrases) in _SLOT_CONFIG.items():
            texts = [phrases[lbl] for lbl in labels]
            tokens = tokenizer(texts).to(device)                # [N, ctx_len]
            embs   = pe_model.encode_text(tokens, normalize=True)  # [N, D]
            slot_embeddings[slot] = (labels, embs.cpu())
    return slot_embeddings


# ---------------------------------------------------------------------------
# Feature extraction for one window
# ---------------------------------------------------------------------------

@torch.no_grad()
def _encode_window(
    crops: List[PILImage.Image],
    pe_model,
    img_transform,
    device: torch.device,
    num_frames: int,
) -> torch.Tensor:
    """
    Uniformly subsample up to num_frames crops, encode each, mean-pool.
    Returns a [D] normalised embedding.
    """
    n = len(crops)
    if n > num_frames:
        idxs = [int(round(i * (n - 1) / (num_frames - 1))) for i in range(num_frames)]
        crops = [crops[i] for i in idxs]
    elif n == 0:
        raise ValueError("No crops provided")

    tensors = torch.stack([img_transform(c) for c in crops]).to(device)  # [N, 3, H, W]
    feats   = pe_model.encode_image(tensors, normalize=False)              # [N, D]
    feat    = feats.mean(dim=0)                                            # [D]
    return F.normalize(feat, dim=-1)


# ---------------------------------------------------------------------------
# Zero-shot classification
# ---------------------------------------------------------------------------

def _classify(
    window_feat: torch.Tensor,
    slot_embeddings: Dict[str, Tuple[List[str], torch.Tensor]],
    top_k: int,
) -> Dict:
    """
    Cosine similarity between window_feat [D] and each slot's label embeddings.
    Returns dict with top-1 label per slot + top-k scores.
    """
    result = {}
    feat = window_feat.cpu()
    for slot, (labels, embs) in slot_embeddings.items():
        sims   = (feat @ embs.T).tolist()                  # [N_labels]
        ranked = sorted(zip(labels, sims), key=lambda x: -x[1])
        result[slot]         = ranked[0][0]                # top-1 label
        result[f"{slot}_topk"] = [[lbl, round(s, 4)] for lbl, s in ranked[:top_k]]
    return result


# ---------------------------------------------------------------------------
# Colour helpers  (shared with pe_description_gen.py style)
# ---------------------------------------------------------------------------

def _identity_color(identity: str) -> Tuple[int, int, int]:
    import cv2
    hue = abs(hash(identity)) % 180
    hsv = np.uint8([[[hue, 210, 230]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def _darken(color: Tuple[int, int, int], factor: float) -> Tuple[int, int, int]:
    return tuple(max(0, int(c * factor)) for c in color)  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Overlay drawing
# ---------------------------------------------------------------------------

def _draw_window_overlay(
    frame: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
    ident: str,
    win: Optional[Dict],
    color: Tuple[int, int, int],
    font_scale: float,
    top_k: int = 1,
) -> None:
    """
    Draw bbox with identity badge and classification labels stacked above.

    Layout (bottom → top, closest to box first):
        safety_event row   ← dark burgundy
        social row         ← darker identity colour
        body_state row     ← identity colour
        obj_verb row       ← darkest, topmost
        ┌[id]── bbox ──────────┐
    """
    import cv2

    font = cv2.FONT_HERSHEY_SIMPLEX
    ft   = 1
    pad  = 3
    MAX_CHARS = 80
    (_, th), baseline = cv2.getTextSize("A", font, font_scale, ft)

    def _draw_row(y_bottom: int, text: str, bg: Tuple) -> int:
        (tw, _), _ = cv2.getTextSize(text, font, font_scale, ft)
        y_top = y_bottom - th - 2 * pad
        cv2.rectangle(frame, (x1, y_top), (x1 + tw + 2 * pad, y_bottom), bg, -1)
        cv2.putText(frame, text, (x1 + pad, y_bottom - pad - baseline),
                    font, font_scale, (255, 255, 255), ft, cv2.LINE_AA)
        return y_top

    # bounding box + identity badge
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    id_tag = f"[{ident}]"
    (tw, _), _ = cv2.getTextSize(id_tag, font, font_scale, ft)
    cv2.rectangle(frame, (x1, y1), (x1 + tw + 2 * pad, y1 + th + 2 * pad), color, -1)
    cv2.putText(frame, id_tag, (x1 + pad, y1 + th + pad - baseline),
                font, font_scale, (255, 255, 255), ft, cv2.LINE_AA)

    if win is None:
        return

    bs  = win.get("body_state", "")
    ov  = win.get("obj_verb", "")
    on_ = win.get("obj_noun", "")
    sc  = win.get("social", "")
    se  = win.get("safety_event", "")

    # Build top-k label strings if requested
    def _topk_str(slot: str) -> str:
        topk = win.get(f"{slot}_topk", [])
        if top_k <= 1 or not topk:
            return ""
        parts = [f"{lbl}({s:.2f})" for lbl, s in topk[:top_k]]
        return "  ".join(parts)

    rows: List[Tuple[str, Tuple]] = []

    # safety_event (closest to box)
    se_str = f"SE:{se}"
    if top_k > 1:
        se_str += f"  {_topk_str('safety_event')}"
    rows.append((se_str[:MAX_CHARS], (55, 45, 45)))

    # social
    sc_str = f"SC:{sc}"
    if top_k > 1:
        sc_str += f"  {_topk_str('social')}"
    rows.append((sc_str[:MAX_CHARS], _darken(color, 0.40)))

    # obj_verb + obj_noun combined
    obj_label = f"OBJ:{ov}→{on_}" if on_ and on_ != "none" else f"OBJ:{ov}"
    if top_k > 1:
        obj_label += f"  {_topk_str('obj_verb')} / {_topk_str('obj_noun')}"
    rows.append((obj_label[:MAX_CHARS], _darken(color, 0.55)))

    # body_state (topmost)
    bs_str = f"BS:{bs}"
    if top_k > 1:
        bs_str += f"  {_topk_str('body_state')}"
    rows.append((bs_str[:MAX_CHARS], color))

    n_rows  = len(rows)
    y_bottom = max(y1, n_rows * (th + 2 * pad) + 4)
    for text, bg in rows:
        y_bottom = _draw_row(y_bottom, text, bg)


# ---------------------------------------------------------------------------
# Window processor
# ---------------------------------------------------------------------------

def _process_window(
    wid: int,
    frame_cache: Dict[int, PILImage.Image],
    bbox_map_w: Dict[str, List[Tuple]],
    pe_model,
    img_transform,
    slot_embeddings: Dict[str, Tuple[List[str], torch.Tensor]],
    device: torch.device,
    args,
    window_frames: int,
    fps: float,
    frame_w: int, frame_h: int,
    max_frames: int,
    identity_windows: Dict[str, Dict[int, Dict]],
) -> None:
    start_fr  = wid * window_frames
    end_fr    = (wid + 1) * window_frames - 1
    start_sec = round(start_fr / fps, 3)
    end_sec   = round((end_fr + 1) / fps, 3)

    print(f"\n  window {wid}  [{start_sec:.1f}s – {end_sec:.1f}s]")

    for ident, bbox_entries in bbox_map_w.items():
        if not bbox_entries:
            continue

        frame_indices = [e[0] for e in bbox_entries]
        bbox_coords   = [(e[1], e[2], e[3], e[4]) for e in bbox_entries]

        # Crop bbox region from each frame
        crops: List[PILImage.Image] = []
        for fidx, (cx, cy, w, h) in zip(frame_indices, bbox_coords):
            if fidx in frame_cache:
                crop = _crop_bbox(frame_cache[fidx], cx, cy, w, h,
                                  args.context_scale, frame_w, frame_h)
                crops.append(crop)

        if not crops:
            print(f"    [{ident}]  no valid crops — skipped")
            continue

        # Encode window → single embedding
        win_feat = _encode_window(crops, pe_model, img_transform, device, args.num_frames)

        # Zero-shot classify
        cls = _classify(win_feat, slot_embeddings, args.top_k)

        print(f"    [{ident}]  BS:{cls['body_state']}  "
              f"OBJ:{cls['obj_verb']}→{cls['obj_noun']}  "
              f"SC:{cls['social']}  SE:{cls['safety_event']}")

        win_result = {
            "start_frame":  start_fr,
            "end_frame":    end_fr,
            "start_sec":    start_sec,
            "end_sec":      end_sec,
            "n_frames":     len(crops),
            "body_state":   cls["body_state"],
            "obj_verb":     cls["obj_verb"],
            "obj_noun":     cls["obj_noun"],
            "social":       cls["social"],
            "safety_event": cls["safety_event"],
            "top_k": {
                "body_state":   cls.get("body_state_topk", []),
                "obj_verb":     cls.get("obj_verb_topk", []),
                "obj_noun":     cls.get("obj_noun_topk", []),
                "social":       cls.get("social_topk", []),
                "safety_event": cls.get("safety_event_topk", []),
            },
        }
        identity_windows[ident][wid] = win_result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = _parse_args()

    try:
        import cv2
    except ImportError:
        sys.exit("opencv-python required.  pip install opencv-python-headless")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------
    # 0. Resolve frame size
    # ------------------------------------------------------------------
    if args.image_size is not None:
        frame_w, frame_h = args.image_size
    else:
        _probe = cv2.VideoCapture(args.video)
        if not _probe.isOpened():
            sys.exit(f"Cannot open video: {args.video}")
        frame_w = int(_probe.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(_probe.get(cv2.CAP_PROP_FRAME_HEIGHT))
        _probe.release()
        if frame_w == 0 or frame_h == 0:
            sys.exit("Could not read frame size. Supply it with --image-size W H.")
    print(f"Frame size: {frame_w}×{frame_h}")

    # ------------------------------------------------------------------
    # 1. Load PE-Core model
    # ------------------------------------------------------------------
    print(f"Loading PE-Core model ({args.model}) …")
    from core.vision_encoder.pe import CLIP
    from core.vision_encoder.tokenizer import SimpleTokenizer
    from core.vision_encoder.transforms import get_image_transform

    pe_model = CLIP.from_config(
        args.model,
        pretrained=not args.no_pretrained,
        checkpoint_path=args.checkpoint,
    ).to(device).eval()

    img_transform = get_image_transform(pe_model.visual.image_size)
    tokenizer     = SimpleTokenizer(context_length=pe_model.context_length)
    print(f"  image size: {pe_model.visual.image_size}px  "
          f"text ctx_len: {pe_model.context_length}  device: {device}")

    # ------------------------------------------------------------------
    # 2. Pre-compute label embeddings
    # ------------------------------------------------------------------
    print("Pre-computing label text embeddings …")
    slot_embeddings = _build_label_embeddings(pe_model, tokenizer, device)
    for slot, (labels, embs) in slot_embeddings.items():
        print(f"  {slot}: {len(labels)} labels  embs={tuple(embs.shape)}")

    # ------------------------------------------------------------------
    # 3. Load tracks
    # ------------------------------------------------------------------
    print(f"Loading tracks from {args.track_file} …")
    tracks    = _load_tracks(args.track_file)
    frame_map = _build_frame_map(tracks)
    print(f"  {len(tracks)} identities, "
          f"{sum(len(v) for v in tracks.values())} track entries")

    # ------------------------------------------------------------------
    # 4. Probe video
    # ------------------------------------------------------------------
    cap_probe = cv2.VideoCapture(args.video)
    if not cap_probe.isOpened():
        sys.exit(f"Cannot open video: {args.video}")
    vid_fps_probe = cap_probe.get(cv2.CAP_PROP_FPS) or 30.0
    vid_total     = int(cap_probe.get(cv2.CAP_PROP_FRAME_COUNT))
    cap_probe.release()

    fps           = args.fps or vid_fps_probe
    window_frames = max(1, int(args.window_sec * fps))
    if args.debug:
        max_frames = int(1 * 60 * fps)
        args.pretty = True
        print("  [debug] clamping to first 1 minute")
    elif args.max_frames is not None:
        max_frames = args.max_frames
    elif args.max_minutes is not None:
        max_frames = int(args.max_minutes * 60 * fps)
    else:
        max_frames = vid_total
    print(f"  fps={fps:.3f}  window={args.window_sec}s ({window_frames} frames)  "
          f"max_frames={max_frames}")

    # ------------------------------------------------------------------
    # 5. Single-pass read — buffer frames per window, classify per window
    # ------------------------------------------------------------------
    identity_windows: Dict[str, Dict[int, Dict]] = {ident: {} for ident in tracks}

    print(f"\nReading {args.video} and classifying with PE-Core …")
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        sys.exit(f"Cannot open video: {args.video}")

    current_wid: int = -1
    win_frame_cache: Dict[int, PILImage.Image] = {}
    win_bbox_map: Dict[str, List] = defaultdict(list)
    frame_idx = 0

    while frame_idx < max_frames:
        ret, bgr = cap.read()
        if not ret:
            break

        wid = frame_idx // window_frames

        if wid != current_wid and current_wid >= 0:
            _process_window(
                current_wid, win_frame_cache, win_bbox_map,
                pe_model, img_transform, slot_embeddings, device, args,
                window_frames, fps, frame_w, frame_h, max_frames,
                identity_windows,
            )
            win_frame_cache = {}
            win_bbox_map    = defaultdict(list)

        current_wid = wid

        if frame_idx in frame_map:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            win_frame_cache[frame_idx] = PILImage.fromarray(rgb)
            for (ident, cx, cy, w, h) in frame_map[frame_idx]:
                win_bbox_map[ident].append((frame_idx, cx, cy, w, h))

        frame_idx += 1
        if frame_idx % 60 == 0:
            print(f"\r  read {frame_idx}/{max_frames} frames", end="", flush=True)

    cap.release()
    print()

    if current_wid >= 0 and win_frame_cache:
        _process_window(
            current_wid, win_frame_cache, win_bbox_map,
            pe_model, img_transform, slot_embeddings, device, args,
            window_frames, fps, frame_w, frame_h, max_frames,
            identity_windows,
        )

    # ------------------------------------------------------------------
    # 6. Build results + write JSON
    # ------------------------------------------------------------------
    results: Dict[str, List[Dict]] = {
        ident: [win for _, win in sorted(identity_windows[ident].items())]
        for ident in tracks
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=(2 if args.pretty else None), ensure_ascii=False)
    print(f"\nSaved JSON → {out_path}  ({len(results)} identities)")

    if args.no_video:
        sys.exit(0)

    # ------------------------------------------------------------------
    # 7. Render annotated video
    # ------------------------------------------------------------------
    print("\nRendering overlay video …")
    frame_anns      = _build_frame_annotations(tracks, max_frames)
    identity_colors = {ident: _identity_color(ident) for ident in tracks}

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        sys.exit(f"Cannot re-open video: {args.video}")

    vid_w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_vid = Path(args.out_video)
    out_vid.parent.mkdir(parents=True, exist_ok=True)
    fourcc  = cv2.VideoWriter_fourcc(*"mp4v")
    writer  = cv2.VideoWriter(str(out_vid), fourcc, fps, (vid_w, vid_h))

    frame_idx = 0
    while frame_idx < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        for ident, x1, y1, x2, y2 in frame_anns.get(frame_idx, []):
            wid   = frame_idx // window_frames
            wins  = identity_windows.get(ident, {})
            if wid not in wins:
                earlier = [k for k in wins if k <= wid]
                wid = max(earlier) if earlier else None  # type: ignore[assignment]
            win_data = wins.get(wid) if wid is not None else None  # type: ignore[arg-type]

            _draw_window_overlay(
                frame, x1, y1, x2, y2,
                ident, win_data, identity_colors[ident],
                args.font_scale, top_k=args.top_k,
            )

        cv2.putText(frame, f"t={frame_idx / fps:.1f}s", (8, vid_h - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

        writer.write(frame)
        frame_idx += 1
        if frame_idx % 60 == 0:
            print(f"\r  render {frame_idx}/{max_frames} frames", end="", flush=True)

    cap.release()
    writer.release()
    print(f"\nSaved video → {out_vid}")
