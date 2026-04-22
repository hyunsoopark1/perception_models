"""
PLM Description Generator + Visualizer  (generative inference)
==============================================================
For each tracked identity, generates a structured description covering:
  • Motion       — what the person's body is doing
  • Social       — how the person relates to others in the scene
  • Activity     — the high-level activity being performed

Each identity is processed as a **video clip** (full frames from the
original video, sampled uniformly) fed to the Perception Language Model (PLM)
with a structured text prompt that includes the identity ID and bounding box
coordinates.  PLM is a *generative* model trained with vision+LLM — we use
it by asking it to answer three questions about the clip and parsing the
free-form text response.

This is fundamentally different from cosine-similarity classification.  The
model sees the full scene and is told which person to focus on, so the output
is specific to each person rather than picking the same "most common" category
bucket across everyone.

Processing loop
---------------
  for each window (stride = window = window_sec, default 6 s):
      for each identity visible in this window:
          frames = full video frames for this window   # (N, 3, H, W) tensor
          prompt = _make_prompt(identity_id)            # asks for Motion/Social/Activity
          response = PLM.generate([(prompt, frames)])
          motion, social, activity = parse(response)
          nearby_ids = spatial_proximity(tracks, identity, window)

Nearby detection
----------------
Uses bbox center distances from the track JSON — no extra video pass needed.
An identity is "nearby" when its center stays within proximity_scale × bbox_avg
for ≥ 30 % of co-visible frames in the same window.

Output JSON
-----------
    {
      "<identity_id>": [
        {
          "start_frame":        <int>,
          "end_frame":          <int>,
          "start_sec":          <float>,
          "end_sec":            <float>,
          "n_frames":           <int>,
          "motion":             "<free-form label>",
          "social_interaction": {
              "label":      "<free-form label>",
              "nearby_ids": ["id_2", "id_5"]
          },
          "activity":           "<free-form label>",
          "raw_response":       "<full PLM response text>",
          "description":        "<verbatim PLM output>"
        },
        ...
      ],
      ...
    }

Usage
-----
    python apps/pe/pe_description_gen.py \\
        --video        input.mp4 \\
        --track-file   tracks.json \\
        --out          descriptions.json \\
        --out-video    overlay.mp4

    # Skip video rendering, process first 600 frames only:
    python apps/pe/pe_description_gen.py \\
        --video        input.mp4 \\
        --track-file   tracks.json \\
        --max-frames   600 \\
        --no-video \\
        --out          descriptions.json

The full video frame is fed to PLM — no bbox cropping.  The bounding box
coordinates are passed in the text prompt so PLM knows which person to focus
on in each frame.
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image as PILImage


# ---------------------------------------------------------------------------
# Prompt builder — per identity, per window
# ---------------------------------------------------------------------------

def _make_prompt(ident: str, nearby_ids: List[str]) -> str:
    """
    Build the PLM prompt for one identity + window.

    Each frame has:
      - A colored rectangle following the target person (ID '{ident}').
      - Grey rectangles on all other tracked people in the vicinity.
    nearby_ids are passed as text so PLM never needs to read image labels.
    """
    if nearby_ids:
        nearby_text = ", ".join(nearby_ids)
        social_instruction = (
            f"Social: <from this list of nearby people [{nearby_text}], "
            f"list only those that this person is physically touching "
            f"(e.g. hugging, holding hands, lifting). "
            f"Use their exact IDs. If none, write 'none'.>"
        )
    else:
        social_instruction = "Social: none"

    return (
        f"Watch this video clip carefully. "
        f"One person is highlighted with a colored rectangle across all frames. "
        f"All other nearby people are shown with grey rectangles. "
        f"Describe only the person inside the colored rectangle. "
        f"Reply in plain English only — do not output coordinates, frame numbers, or timestamps. "
        f"Use this exact format:\n"
        f"Motion: <how this person is moving, in 2-5 words>\n"
        f"{social_instruction}\n"
        f"Activity: <what this person is doing, in 2-5 words>\n"
        f"If a field is not clearly visible write 'unclear'."
    )


def _make_attn_bias_prompt(ident: str, nearby_ids: List[str]) -> str:
    """
    Prompt for attention-bias inference mode.
    No visual annotation is drawn on the frames; PLM's attention is steered
    to the target person's patch region via bias injection.
    """
    if nearby_ids:
        nearby_text = ", ".join(nearby_ids)
        social_instruction = (
            f"Social: <from this list of nearby people [{nearby_text}], "
            f"list only those that this person is physically touching "
            f"(e.g. hugging, holding hands, lifting). "
            f"Use their exact IDs. If none, write 'none'.>"
        )
    else:
        social_instruction = "Social: none"

    return (
        f"Watch this video clip carefully. "
        f"Focus on the highlighted person in the scene and describe only them. "
        f"Reply in plain English only — do not output coordinates, frame numbers, or timestamps. "
        f"Use this exact format:\n"
        f"Motion: <how this person is moving, in 2-5 words>\n"
        f"{social_instruction}\n"
        f"Activity: <what this person is doing, in 2-5 words>\n"
        f"If a field is not clearly visible write 'unclear'."
    )


def _make_attn_bias_taxonomy_prompt(ident: str, nearby_ids: List[str]) -> str:
    """Taxonomy prompt for attention-bias mode — no visual annotation on frames."""
    nearby_text = ", ".join(nearby_ids) if nearby_ids else "none"
    bs_list = " | ".join(sorted(BODY_STATES))
    ov_list = " | ".join(sorted(OBJ_VERBS))
    on_list = " | ".join(sorted(OBJ_NOUNS_CORE)) + " | <other object if needed>"
    sc_list = " | ".join(sorted(SOCIAL_TAX))
    se_list = " | ".join(sorted(SAFETY_EVENTS))
    return (
        f"Watch this video clip. Focus on the highlighted person in the scene.\n"
        f"Nearby people: {nearby_text}\n\n"
        f"Classify ONLY the highlighted person. Pick exactly one label per slot:\n\n"
        f"body_state:   {bs_list}\n"
        f"obj_verb:     {ov_list}\n"
        f"obj_noun:     {on_list}\n"
        f"social:       {sc_list}\n"
        f"              REQUIRED when not none: also list which nearby person ID(s)\n"
        f"              are involved, e.g.  social: co_manipulate [d14709]\n"
        f"safety_event: {se_list}\n"
        f"other_text:   (free text only if none of the slots above apply; else leave blank)\n\n"
        f"Reply ONLY in this format, one field per line:\n"
        f"body_state: <label>\n"
        f"obj_verb: <label>\n"
        f"obj_noun: <label>\n"
        f"social: <label> [<id1>, <id2>]  — or —  social: none\n"
        f"safety_event: <label>\n"
        f"other_text: <text or blank>"
    )


# ---------------------------------------------------------------------------
# Taxonomy — structured classification  (separate PLM call from M/S/A)
# ---------------------------------------------------------------------------

BODY_STATES = frozenset({
    "idle_stand", "idle_sit", "walk", "walk_loaded", "run",
    "bend", "squat", "kneel", "reach_overhead", "reach_low",
    "twist", "crouch_sustained", "climb", "fall", "recover_balance",
})
OBJ_VERBS = frozenset({
    "reach", "grasp", "lift", "lower", "carry", "place", "push",
    "pull", "drag", "stack", "unstack", "pack", "unpack", "scan",
    "inspect", "operate", "throw", "catch", "rotate", "none",
})
OBJ_NOUNS_CORE = frozenset({
    "box", "pallet", "scanner", "cart", "forklift", "ladder",
    "tool", "document", "shelf", "bin", "bag", "package",
    "button", "screen", "door", "handle", "none",
})
SOCIAL_TAX = frozenset({
    "none", "talk", "handover", "receive", "co_manipulate",
    "gesture_instruct", "point",
})
SAFETY_EVENTS = frozenset({
    "none", "zone_enter", "zone_exit", "ppe_don", "ppe_doff",
    "near_miss", "hazard_response", "fall",
})


def _make_taxonomy_prompt(ident: str, nearby_ids: List[str]) -> str:
    nearby_text = ", ".join(nearby_ids) if nearby_ids else "none"
    bs_list = " | ".join(sorted(BODY_STATES))
    ov_list = " | ".join(sorted(OBJ_VERBS))
    on_list = " | ".join(sorted(OBJ_NOUNS_CORE)) + " | <other object if needed>"
    sc_list = " | ".join(sorted(SOCIAL_TAX))
    se_list = " | ".join(sorted(SAFETY_EVENTS))
    return (
        f"Watch this video clip. One person is highlighted with a colored rectangle.\n"
        f"Nearby people: {nearby_text}\n\n"
        f"Classify ONLY the highlighted person. Pick exactly one label per slot:\n\n"
        f"body_state:   {bs_list}\n"
        f"obj_verb:     {ov_list}\n"
        f"obj_noun:     {on_list}\n"
        f"social:       {sc_list}\n"
        f"              REQUIRED when not none: also list which nearby person ID(s)\n"
        f"              are involved, e.g.  social: co_manipulate [d14709]\n"
        f"safety_event: {se_list}\n"
        f"other_text:   (free text only if none of the slots above apply; else leave blank)\n\n"
        f"Reply ONLY in this format, one field per line:\n"
        f"body_state: <label>\n"
        f"obj_verb: <label>\n"
        f"obj_noun: <label>\n"
        f"social: <label> [<id1>, <id2>]  — or —  social: none\n"
        f"safety_event: <label>\n"
        f"other_text: <text or blank>"
    )


def _match_label(val: str, allowed: frozenset, default: str) -> str:
    """Fuzzy match PLM output against allowed label set."""
    v = val.lower().strip()
    if v in allowed:
        return v
    v_norm = re.sub(r"[\s\-]+", "_", v)
    if v_norm in allowed:
        return v_norm
    for a in sorted(allowed):
        if a in v or v in a:
            return a
    return default


def _parse_taxonomy_response(text: str, nearby_ids: Optional[List[str]] = None) -> Dict:
    """
    Parse structured taxonomy response; validate each slot against its allowed set.
    The 'social' slot becomes {"label": str, "with_ids": list[str]}.
    If PLM omits IDs for a non-none social label, nearby_ids is used as fallback
    because any social interaction must involve a co-present person.
    """
    result: Dict = {
        "body_state":   "unknown",
        "obj_verb":     "none",
        "obj_noun":     "none",
        "social":       {"label": "none", "with_ids": []},
        "safety_event": "none",
        "other_text":   "",
    }
    slot_cfg = {
        "body_state":   (BODY_STATES,   "unknown"),
        "obj_verb":     (OBJ_VERBS,     "none"),
        "obj_noun":     (None,          "none"),   # free-form noun accepted
        "social":       (SOCIAL_TAX,    "none"),   # special-cased below
        "safety_event": (SAFETY_EVENTS, "none"),
        "other_text":   (None,          ""),
    }
    for line in text.splitlines():
        stripped = line.strip().replace("**", "")
        low = stripped.lower()
        for key in slot_cfg:
            m = re.match(rf"^{re.escape(key)}\s*[:–\-]\s*", low)
            if m:
                val = stripped[m.end():].strip()
                allowed, default = slot_cfg[key]
                if key == "social":
                    # Extract IDs from brackets, e.g. "co_manipulate [d14709]"
                    with_ids = re.findall(r'\b[a-zA-Z]\d{4,}\b', val)
                    label_part = re.sub(r'\[.*?\]', '', val).strip()
                    label = _match_label(label_part, SOCIAL_TAX, "none")
                    # Fallback: any non-none interaction must involve someone —
                    # use nearby_ids when PLM forgot to include brackets.
                    if label != "none" and not with_ids and nearby_ids:
                        with_ids = list(nearby_ids)
                    result["social"] = {"label": label, "with_ids": with_ids}
                elif allowed is not None:
                    result[key] = _match_label(val, allowed, default)
                else:
                    result[key] = val if val else default
                break
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="PLM generative description per identity per 2-sec window + video overlay.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # --- inputs ---
    p.add_argument("--video", required=True, metavar="PATH",
                   help="Input video file.")
    p.add_argument("--track-file", required=True, metavar="FILE",
                   help="Identity-format JSON: {id: [[frame,cx,cy,w,h], ...], ...}")
    p.add_argument("--image-size", default=None, type=int, nargs=2, metavar=("W", "H"),
                   help="Original frame size in pixels (W H). "
                        "Inferred from the video file when omitted.")
    # --- model ---
    p.add_argument("--plm-ckpt", default="facebook/Perception-LM-8B", metavar="CKPT",
                   help="PLM checkpoint or HF model ID (default: facebook/Perception-LM-8B).")
    p.add_argument("--num-plm-frames", type=int, default=16, metavar="N",
                   help="Frames sampled per 2-sec clip for PLM (default: 16).")
    p.add_argument("--max-gen-len", type=int, default=120, metavar="N",
                   help="Max generated tokens per clip (default: 120).")
    p.add_argument("--temperature", type=float, default=0.0,
                   help="Sampling temperature; 0 = greedy (default: 0).")
    # --- windowing ---
    p.add_argument("--window-sec", type=float, default=6.0, metavar="S",
                   help="Clip duration in seconds (default: 6.0).")
    p.add_argument("--fps", type=float, default=None, metavar="N",
                   help="Override video FPS (inferred from file when omitted).")
    # --- runtime ---
    p.add_argument("--max-frames", type=int, default=None, metavar="N",
                   help="Stop after this many frames (default: all).")
    p.add_argument("--max-minutes", type=float, default=None, metavar="M",
                   help="Stop after this many minutes of video (default: all). "
                        "Overridden by --max-frames if both are given.")
    # --- attention bias mode ---
    p.add_argument("--attn-bias", action="store_true",
                   help="Use attention bias instead of drawing boxes on frames. "
                        "The image is unmodified; PLM attention is steered toward "
                        "the target person's patch region via SDPA bias injection.")
    p.add_argument("--bbox-bias", type=float, default=3.0, metavar="B",
                   help="Additive logit bias for bbox patches in --attn-bias mode "
                        "(default: 3.0 ≈ 20× relative attention weight).")
    # --- proximity ---
    p.add_argument("--proximity-scale", type=float, default=2.0, metavar="S",
                   help="Nearby threshold = this × avg_bbox_dim (default: 2.0).")
    # --- output ---
    p.add_argument("--out", default="descriptions.json", metavar="PATH",
                   help="Output JSON path (default: descriptions.json).")
    p.add_argument("--out-video", default="description_overlay.mp4", metavar="PATH",
                   help="Output annotated video path (default: description_overlay.mp4).")
    p.add_argument("--no-msa", action="store_true",
                   help="Skip the Motion/Social/Activity PLM call; run taxonomy only.")
    p.add_argument("--no-video", action="store_true",
                   help="Skip video rendering.")
    p.add_argument("--compare", action="store_true",
                   help="Run both default (colored-box) and attn-bias modes on every "
                        "clip and render both result sets on the overlay video side-by-side "
                        "for direct comparison. Implies --attn-bias is available.")
    p.add_argument("--font-scale", type=float, default=0.42,
                   help="cv2 font scale for overlay labels (default: 0.42).")
    p.add_argument("--pretty", action="store_true",
                   help="Pretty-print JSON (indent=2).")
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
# Proximity detection  (track data only, no video re-read)
# ---------------------------------------------------------------------------

def _find_nearby_ids(
    ident: str,
    wid: int,
    window_frames: int,
    tracks: Dict[str, List],
    proximity_scale: float,
    max_frames: int,
    min_close_ratio: float = 0.30,
) -> List[str]:
    frame_start = wid * window_frames
    frame_end   = (wid + 1) * window_frames

    target: Dict[int, Tuple] = {}
    for entry in tracks[ident]:
        fidx = int(entry[0])
        if frame_start <= fidx < frame_end and fidx < max_frames:
            target[fidx] = (float(entry[1]), float(entry[2]),
                            float(entry[3]), float(entry[4]))
    if not target:
        return []

    nearby: List[str] = []
    for other_id, other_entries in tracks.items():
        if other_id == ident:
            continue
        other: Dict[int, Tuple] = {}
        for entry in other_entries:
            fidx = int(entry[0])
            if fidx in target:
                other[fidx] = (float(entry[1]), float(entry[2]))

        co_visible = set(target.keys()) & set(other.keys())
        if not co_visible:
            continue

        close_count = sum(
            1 for fidx in co_visible
            if (
                (target[fidx][0] - other[fidx][0]) ** 2 +
                (target[fidx][1] - other[fidx][1]) ** 2
            ) ** 0.5 < proximity_scale * (target[fidx][2] + target[fidx][3]) / 2.0
        )
        if close_count / len(co_visible) >= min_close_ratio:
            nearby.append(other_id)

    return sorted(nearby)


# ---------------------------------------------------------------------------
# PLM response parsing
# ---------------------------------------------------------------------------

def _parse_plm_response(text: str) -> Tuple[str, str, str]:
    """
    Extract Motion / Social / Activity from the PLM's response.

    Robust to common PLM formatting variations:
      - "Motion: sitting still"
      - "**Motion:** sitting still"   (markdown bold)
      - "Motion - sitting still"      (dash separator)
      - lowercase / mixed case
    Returns empty string for any field the model didn't fill in.
    """
    motion = social = activity = ""
    for line in text.splitlines():
        # Strip markdown bold markers before matching
        stripped = line.strip().replace("**", "")
        low = stripped.lower()
        for prefix, field in (
            ("motion",   "motion"),
            ("social",   "social"),
            ("activity", "activity"),
        ):
            m = re.match(rf"^{prefix}\s*[:–\-]\s*", low)
            if m:
                val = stripped[m.end():].strip()
                # Strip angle brackets if PLM answered inside <...>;
                # discard only if it still looks like an unfilled template keyword.
                if re.fullmatch(r"<[^>]*>", val):
                    inner = val[1:-1].strip()
                    val = "" if re.search(r"\b(how|what|IDs?|e\.g\.)\b", inner, re.I) else inner
                if field == "motion":
                    motion = val
                elif field == "social":
                    social = val
                elif field == "activity":
                    activity = val
                break
    return motion, social, activity


# ---------------------------------------------------------------------------
# Description composer
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Per-frame bbox annotation
# ---------------------------------------------------------------------------

def _annotate_frame_base(
    pil_frame: PILImage.Image,
    all_bboxes: Dict[str, Tuple[float, float, float, float]],
) -> np.ndarray:
    """
    Draw every tracked person's bbox in grey with their ID label.
    Returns a numpy array (not PIL) so _annotate_frame_target can cheaply
    copy it and add one colored box without re-drawing all grey boxes.
    Called once per unique frame per window, shared across all identities.
    """
    import cv2
    frame = np.array(pil_frame)
    H, W  = frame.shape[:2]
    for oid, (cx, cy, w, h) in all_bboxes.items():
        x1 = max(0, int(cx - w / 2));  y1 = max(0, int(cy - h / 2))
        x2 = min(W, int(cx + w / 2));  y2 = min(H, int(cy + h / 2))
        cv2.rectangle(frame, (x1, y1), (x2, y2), (180, 180, 180), 2)
        cv2.putText(frame, oid, (x1 + 3, y1 + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1, cv2.LINE_AA)
    return frame


def _annotate_frame_target(
    base_frame: np.ndarray,
    cx: float, cy: float, w: float, h: float,
    color: Tuple[int, int, int],
    thickness: int = 4,
) -> PILImage.Image:
    """
    Copy the pre-annotated base frame and add the target's colored box.
    Only this function is called per-identity; the grey base is shared.
    """
    import cv2
    frame = base_frame.copy()
    H, W  = frame.shape[:2]
    x1 = max(0, int(cx - w / 2));  y1 = max(0, int(cy - h / 2))
    x2 = min(W, int(cx + w / 2));  y2 = min(H, int(cy + h / 2))
    bgr = (color[2], color[1], color[0])
    cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, thickness)
    return PILImage.fromarray(frame)


# ---------------------------------------------------------------------------
# Colour helpers
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
# Visualization drawing
# ---------------------------------------------------------------------------

def _draw_window_overlay(
    frame: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
    ident: str,
    win: Optional[Dict],
    color: Tuple[int, int, int],
    font_scale: float,
) -> None:
    """
    Draw bbox with identity badge and label rows stacked above it (bottom → top).

    Normal mode:
        BS:<body_state>  OBJ:<verb>→<noun>   ← taxonomy (topmost, dark slate)
        SC:<social_tax>  SE:<safety_event>   ← taxonomy (dark burgundy)
        M: <motion>                          ← identity color
        S: <social_desc>                     ← darker
        A: <activity>                        ← darkest, closest to box
        ┌[id]──── bbox ───────────────────┐

    Compare mode (win["compare_attn_bias"] present):
        AB·BS:...  OBJ:...                   ← AB taxonomy  (teal-slate, topmost)
        AB·SC:...  SE:...                    ← AB taxonomy
        AB▸ M:...  A:...  S:...             ← AB M/S/A     (steel blue)
        ─────────────────────────────────── ← thin separator
        BS:...  OBJ:...                      ← DEF taxonomy (dark slate)
        SC:...  SE:...                       ← DEF taxonomy
        M: <default motion>                  ← DEF M/S/A   (identity color)
        S: <default social>
        A: <default activity>
        ┌[id]──── bbox ───────────────────┐
    """
    import cv2

    font = cv2.FONT_HERSHEY_SIMPLEX
    ft   = 1
    pad  = 3
    MAX_CHARS = 72
    (_, th), baseline = cv2.getTextSize("A", font, font_scale, ft)

    def _draw_row(y_bottom: int, text: str, bg: Tuple,
                  text_color: Tuple = (255, 255, 255)) -> int:
        (tw, _), _ = cv2.getTextSize(text, font, font_scale, ft)
        y_top = y_bottom - th - 2 * pad
        cv2.rectangle(frame, (x1, y_top), (x1 + tw + 2 * pad, y_bottom), bg, -1)
        cv2.putText(frame, text, (x1 + pad, y_bottom - pad - baseline),
                    font, font_scale, text_color, ft, cv2.LINE_AA)
        return y_top

    def _draw_separator(y_bottom: int) -> int:
        """Thin 1-px rule to visually separate compare groups."""
        cv2.line(frame, (x1, y_bottom - 1), (x1 + 180, y_bottom - 1),
                 (180, 180, 180), 1)
        return y_bottom - 2

    # --- bounding box ---
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # --- identity badge ---
    id_tag = f"[{ident}]"
    (tw, _), _ = cv2.getTextSize(id_tag, font, font_scale, ft)
    cv2.rectangle(frame, (x1, y1), (x1 + tw + 2 * pad, y1 + th + 2 * pad), color, -1)
    cv2.putText(frame, id_tag, (x1 + pad, y1 + th + pad - baseline),
                font, font_scale, (255, 255, 255), ft, cv2.LINE_AA)

    if win is None:
        return

    # --- collect field values ---
    motion      = win.get("motion", "")
    social_dict = win.get("social_interaction", {})
    activity    = win.get("activity", "")
    social_lbl  = (social_dict.get("label", "")
                   if isinstance(social_dict, dict) else str(social_dict))

    compare = win.get("compare_attn_bias")  # present only in --compare mode

    # Build rows list bottom → top (index 0 = closest to box).
    rows: List[Tuple] = []  # (text, bg)  or  ("__SEP__", None) for separator

    def _tax_to_rows(tax: Dict, prefix: str = "",
                     bg_sc_se=(55, 45, 45), bg_bs_obj=(45, 45, 65)) -> None:
        """Append two taxonomy display rows (SC/SE then BS/OBJ) to `rows`."""
        bs      = tax.get("body_state", "")
        ov      = tax.get("obj_verb", "none")
        on_     = tax.get("obj_noun", "none")
        sc_dict = tax.get("social", {"label": "none", "with_ids": []})
        se      = tax.get("safety_event", "none")
        ot      = tax.get("other_text", "")
        if isinstance(sc_dict, dict):
            sc_label = sc_dict.get("label", "none")
            sc_with  = sc_dict.get("with_ids", [])
        else:
            sc_label, sc_with = str(sc_dict), []
        sc_str = (f"SC:{sc_label}[{','.join(sc_with)}]" if sc_label != "none" else "")
        se_str = f"SE:{se}" if se != "none" else ""
        ot_str = f"[{ot[:28]}]" if ot else ""
        row_sc_se = "  ".join(filter(None, [sc_str, se_str, ot_str])) or "SC:none  SE:none"
        rows.append(((prefix + row_sc_se)[:MAX_CHARS], bg_sc_se, None))
        obj_part = (f"OBJ:{ov}→{on_}" if ov != "none" and on_ != "none"
                    else f"OBJ:{ov}" if ov != "none"
                    else f"OBJ:{on_}" if on_ != "none" else "")
        row_bs = f"{prefix}BS:{bs}" + (f"  {obj_part}" if obj_part else "")
        rows.append((row_bs[:MAX_CHARS], bg_bs_obj, None))

    if compare:
        # ----------------------------------------------------------------
        # Compare mode layout (bottom → top):
        #   DEF M/S/A rows  (identity color)
        #   DEF taxonomy    (dark slate)
        #   ─── separator ──
        #   AB▸ M/S/A row   (steel blue)
        #   AB taxonomy     (lighter teal-slate, topmost)
        # ----------------------------------------------------------------
        if activity:
            rows.append((f"A: {activity[:MAX_CHARS]}", _darken(color, 0.38), None))
        if social_lbl:
            rows.append((f"S: {social_lbl[:MAX_CHARS]}", _darken(color, 0.60), None))
        if motion:
            rows.append((f"M: {motion[:MAX_CHARS]}", color, None))

        taxonomy = win.get("taxonomy", {})
        if taxonomy:
            _tax_to_rows(taxonomy, prefix="", bg_sc_se=(55, 45, 45), bg_bs_obj=(45, 45, 65))

        rows.append(("__SEP__", None, None))

        # AB M/S/A (condensed into one row)
        ab_m = compare.get("motion", "")
        ab_a = compare.get("activity", "")
        ab_s_dict = compare.get("social", {})
        ab_s = (ab_s_dict.get("label", "") if isinstance(ab_s_dict, dict)
                else str(ab_s_dict)) if ab_s_dict else ""
        ab_parts = []
        if ab_m:
            ab_parts.append(f"M:{ab_m}")
        if ab_a:
            ab_parts.append(f"A:{ab_a}")
        if ab_s:
            ab_parts.append(f"S:{ab_s}")
        if ab_parts:
            rows.append((("AB▸ " + "  ".join(ab_parts))[:MAX_CHARS], (80, 60, 30), None))

        # AB taxonomy (lighter teal-slate so it's visually distinct from DEF taxonomy)
        ab_taxonomy = compare.get("taxonomy", {})
        if ab_taxonomy:
            _tax_to_rows(ab_taxonomy, prefix="AB·",
                         bg_sc_se=(85, 65, 50), bg_bs_obj=(65, 65, 90))

    else:
        # ----------------------------------------------------------------
        # Normal mode layout (bottom → top):
        #   M/S/A rows  (identity color)
        #   taxonomy    (dark slate, topmost)
        # ----------------------------------------------------------------
        if activity:
            rows.append((f"A: {activity[:MAX_CHARS]}", _darken(color, 0.38), None))
        if social_lbl:
            rows.append((f"S: {social_lbl[:MAX_CHARS]}", _darken(color, 0.60), None))
        if motion:
            rows.append((f"M: {motion[:MAX_CHARS]}", color, None))

        if not (motion or social_lbl or activity):
            raw = win.get("description", "")
            if raw:
                rows.append((raw[:MAX_CHARS], _darken(color, 0.60), None))

        taxonomy = win.get("taxonomy", {})
        if taxonomy:
            _tax_to_rows(taxonomy, prefix="", bg_sc_se=(55, 45, 45), bg_bs_obj=(45, 45, 65))

    n_real_rows = sum(1 for r in rows if r[0] != "__SEP__")
    y_bottom = max(y1, n_real_rows * (th + 2 * pad) + len(rows) + 4)
    for text, bg, _ in rows:
        if text == "__SEP__":
            y_bottom = _draw_separator(y_bottom)
        else:
            y_bottom = _draw_row(y_bottom, text, bg)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = _parse_args()

    try:
        import cv2
    except ImportError:
        sys.exit("opencv-python required.  pip install opencv-python-headless")

    # ------------------------------------------------------------------
    # 0. Resolve frame size (auto-detect from video when not supplied)
    # ------------------------------------------------------------------
    if args.image_size is not None:
        frame_w, frame_h = args.image_size
    else:
        _probe = cv2.VideoCapture(args.video)
        if not _probe.isOpened():
            sys.exit(f"Cannot open video to detect resolution: {args.video}")
        frame_w = int(_probe.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(_probe.get(cv2.CAP_PROP_FRAME_HEIGHT))
        _probe.release()
        if frame_w == 0 or frame_h == 0:
            sys.exit("Could not read frame size from video. "
                     "Supply it manually with --image-size W H.")
    print(f"Frame size: {frame_w}×{frame_h}")

    # ------------------------------------------------------------------
    # 1. Load PLM + set up generator
    # ------------------------------------------------------------------
    print(f"Loading PLM from {args.plm_ckpt} …")
    from apps.plm.generate import (
        PackedCausalTransformerGenerator,
        PackedCausalTransformerGeneratorArgs,
        load_consolidated_model_and_tokenizer,
    )
    from core.transforms.video_transform import get_video_transform

    plm_model, plm_tokenizer, plm_config = load_consolidated_model_and_tokenizer(
        args.plm_ckpt
    )
    plm_transform = get_video_transform(image_res=plm_model.vision_model.image_size)
    print(f"  vision input size: {plm_model.vision_model.image_size}px")

    gen_cfg = PackedCausalTransformerGeneratorArgs(
        temperature=args.temperature,
        max_gen_len=args.max_gen_len,
        until=[],           # let the model generate all 3 lines
        dtype="bf16",
        device="cuda",
    )
    generator = PackedCausalTransformerGenerator(gen_cfg, plm_model, plm_tokenizer)

    # Patch grid parameters (needed for attention-bias mode)
    _vis_image_size  = plm_model.vision_model.image_size
    _tok_patch_size  = getattr(plm_tokenizer, "patch_size",  14)
    _tok_pool_ratio  = getattr(plm_tokenizer, "pooling_ratio", 1)
    _n_patches_side  = _vis_image_size // _tok_patch_size // _tok_pool_ratio
    _patches_per_frm = _n_patches_side ** 2

    if args.attn_bias or args.compare:
        from apps.pe.pe_attn_bias import (
            bbox_attention_bias,
            compute_bbox_bias_mask,
            get_image_patch_positions,
        )
        print(f"  attn-bias mode: patch_grid={_n_patches_side}×{_n_patches_side}  "
              f"patches/frame={_patches_per_frm}  bbox_bias={args.bbox_bias}")

    print(f"  generator ready  (max_gen_len={args.max_gen_len}, "
          f"temperature={args.temperature})")
    mode_desc = "attention-bias (no frame drawing)" if args.attn_bias else "bbox overlay (colored rectangle)"
    print(f"  mode: {mode_desc}\n")

    # ------------------------------------------------------------------
    # 2. Load tracks
    # ------------------------------------------------------------------
    print(f"Loading tracks from {args.track_file} …")
    tracks    = _load_tracks(args.track_file)
    frame_map = _build_frame_map(tracks)
    print(f"  {len(tracks)} identities, "
          f"{sum(len(v) for v in tracks.values())} track entries")

    # ------------------------------------------------------------------
    # 3. Probe video
    # ------------------------------------------------------------------
    cap_probe = cv2.VideoCapture(args.video)
    if not cap_probe.isOpened():
        sys.exit(f"Cannot open video: {args.video}")
    vid_fps_probe = cap_probe.get(cv2.CAP_PROP_FPS) or 30.0
    vid_total     = int(cap_probe.get(cv2.CAP_PROP_FRAME_COUNT))
    cap_probe.release()

    fps           = args.fps or vid_fps_probe
    window_frames = max(1, int(args.window_sec * fps))
    if args.max_frames is not None:
        max_frames = args.max_frames
    elif args.max_minutes is not None:
        max_frames = int(args.max_minutes * 60 * fps)
    else:
        max_frames = vid_total
    print(f"  fps={fps:.3f}  window={args.window_sec}s ({window_frames} frames)  "
          f"max_frames={max_frames}")

    # ------------------------------------------------------------------
    # 4. Single-pass read — buffer full frames per window, generate per window
    #
    # Each unique video frame is stored once (shared across all identities
    # visible in that frame).  Per-identity we only keep bbox coordinates.
    # ------------------------------------------------------------------
    # identity → {wid: result_dict}
    identity_windows: Dict[str, Dict[int, Dict]] = {ident: {} for ident in tracks}

    def _process_window(
        wid: int,
        frame_cache: Dict[int, PILImage.Image],          # frame_idx → full PIL frame
        bbox_map_w:  Dict[str, List[Tuple]],             # identity → [(fidx,cx,cy,w,h)]
    ) -> None:
        """Run PLM generation for every identity visible in window *wid*."""
        start_fr  = wid * window_frames
        end_fr    = (wid + 1) * window_frames - 1
        start_sec = round(start_fr / fps, 3)
        end_sec   = round((end_fr + 1) / fps, 3)

        print(f"\n  window {wid}  [{start_sec:.1f}s – {end_sec:.1f}s]")

        for ident, bbox_entries in bbox_map_w.items():
            if not bbox_entries:
                continue

            # Collect full frames (in frame order, from the shared cache)
            frame_indices = [e[0] for e in bbox_entries]
            pil_frames    = [frame_cache[fidx] for fidx in frame_indices]
            bbox_coords   = [(e[1], e[2], e[3], e[4]) for e in bbox_entries]

            # Uniform subsample — keep frame_indices and bbox_coords in sync
            n = len(pil_frames)
            if n > args.num_plm_frames:
                idxs = [int(round(i * (n - 1) / (args.num_plm_frames - 1)))
                        for i in range(args.num_plm_frames)]
                frame_indices = [frame_indices[i] for i in idxs]
                pil_frames    = [pil_frames[i]    for i in idxs]
                bbox_coords   = [bbox_coords[i]   for i in idxs]
            elif n == 1:
                frame_indices = frame_indices * 2
                pil_frames    = pil_frames    * 2
                bbox_coords   = bbox_coords   * 2

            # Build per-frame bbox lookup for all other identities in this window
            # so their boxes + IDs can be drawn onto each frame for PLM to read.
            other_frame_bboxes: Dict[int, Dict[str, Tuple]] = defaultdict(dict)
            for other_id, other_entries in bbox_map_w.items():
                if other_id == ident:
                    continue
                for (fidx, cx, cy, w, h) in other_entries:
                    other_frame_bboxes[fidx][other_id] = (cx, cy, w, h)

            nearby_ids = _find_nearby_ids(
                ident, wid, window_frames, tracks,
                args.proximity_scale, max_frames,
            )

            # ----------------------------------------------------------
            # Build frame tensors (done once, shared across helpers)
            # ----------------------------------------------------------
            ann_color = _identity_color(ident)
            annotated = [
                _annotate_frame_target(
                    _annotate_frame_base(frame_cache[fidx],
                                         other_frame_bboxes.get(fidx, {})),
                    cx, cy, w, h, ann_color,
                )
                for fidx, (cx, cy, w, h) in zip(frame_indices, bbox_coords)
            ]
            frames_ann, _ = plm_transform._process_multiple_images_pil(annotated)
            frames_raw, _ = plm_transform._process_multiple_images_pil(pil_frames)

            # ----------------------------------------------------------
            # Helpers: one attn-bias M/S/A call, one attn-bias taxonomy call
            # ----------------------------------------------------------
            def _ab_bias_mask(prompt_str, ft):
                ip, sl = get_image_patch_positions(generator.tokenizer, prompt_str, ft)
                return compute_bbox_bias_mask(
                    ip, bbox_coords,
                    patches_per_frame=_patches_per_frm,
                    n_patches_side=_n_patches_side,
                    orig_w=frame_w, orig_h=frame_h,
                    image_size=_vis_image_size,
                    seq_len=sl, bias=args.bbox_bias,
                )

            def _run_ab_msa(ft):
                p = _make_attn_bias_prompt(ident, nearby_ids)
                with bbox_attention_bias(_ab_bias_mask(p, ft)):
                    r, _, _ = generator.generate([(p, ft)])
                raw_ = r[0].strip()
                m, s, a = _parse_plm_response(raw_)
                print(f"      [AB] M:{m!r}  S:{s!r}  A:{a!r}")
                print(f"      [AB] raw → {raw_!r}")
                return m, s, a, raw_

            def _run_ab_taxonomy(ft):
                p = _make_attn_bias_taxonomy_prompt(ident, nearby_ids)
                with bbox_attention_bias(_ab_bias_mask(p, ft)):
                    r, _, _ = generator.generate([(p, ft)])
                raw_ = r[0].strip()
                tax_ = _parse_taxonomy_response(raw_, nearby_ids)
                sc_d_ = tax_['social']
                sc_s_ = (f"{sc_d_['label']} {sc_d_['with_ids']}"
                         if isinstance(sc_d_, dict) else str(sc_d_))
                print(f"      [AB] taxonomy → BS:{tax_['body_state']}  "
                      f"OV:{tax_['obj_verb']}  ON:{tax_['obj_noun']}  "
                      f"SC:{sc_s_}  SE:{tax_['safety_event']}")
                return tax_

            # ----------------------------------------------------------
            # Dispatch by mode
            # ----------------------------------------------------------
            if args.compare:
                # Default M/S/A
                def_p = _make_prompt(ident, nearby_ids)
                def_r, _, _ = generator.generate([(def_p, frames_ann)])
                raw = def_r[0].strip()
                motion, social, activity = _parse_plm_response(raw)
                print(f"    [{ident}] [DEF] M:{motion!r}  S:{social!r}  A:{activity!r}")

                # Default taxonomy
                tax_p = _make_taxonomy_prompt(ident, nearby_ids)
                tax_r, _, _ = generator.generate([(tax_p, frames_ann)])
                tax_raw = tax_r[0].strip()
                taxonomy = _parse_taxonomy_response(tax_raw, nearby_ids)

                # Attn-bias M/S/A + taxonomy
                ab_motion, ab_social, ab_activity, ab_raw = _run_ab_msa(frames_raw)
                ab_taxonomy = _run_ab_taxonomy(frames_raw)

                compare_ab = {
                    "motion":    ab_motion,
                    "social":    ab_social,
                    "activity":  ab_activity,
                    "taxonomy":  ab_taxonomy,
                    "description": ab_raw,
                }

            elif args.attn_bias:
                if not args.no_msa:
                    motion, social, activity, raw = _run_ab_msa(frames_raw)
                else:
                    motion = social = activity = raw = ""
                taxonomy  = _run_ab_taxonomy(frames_raw)
                compare_ab = None

            else:
                # Default only
                def_p = _make_prompt(ident, nearby_ids)
                if not args.no_msa:
                    def_r, _, _ = generator.generate([(def_p, frames_ann)])
                    raw = def_r[0].strip()
                    motion, social, activity = _parse_plm_response(raw)
                    print(f"    [{ident}]  M:{motion!r}  S:{social!r}  A:{activity!r}")
                    print(f"      raw → {raw!r}")
                else:
                    motion = social = activity = raw = ""

                tax_p = _make_taxonomy_prompt(ident, nearby_ids)
                tax_r, _, _ = generator.generate([(tax_p, frames_ann)])
                tax_raw = tax_r[0].strip()
                taxonomy = _parse_taxonomy_response(tax_raw, nearby_ids)
                compare_ab = None

            sc_d = taxonomy['social']
            sc_str = (f"{sc_d['label']} {sc_d['with_ids']}"
                      if isinstance(sc_d, dict) else str(sc_d))
            print(f"    [{ident}]  taxonomy → BS:{taxonomy['body_state']}  "
                  f"OV:{taxonomy['obj_verb']}  ON:{taxonomy['obj_noun']}  "
                  f"SC:{sc_str}  SE:{taxonomy['safety_event']}"
                  + (f"  other:{taxonomy['other_text']!r}" if taxonomy['other_text'] else ""))

            win = {
                "start_frame":        start_fr,
                "end_frame":          end_fr,
                "start_sec":          start_sec,
                "end_sec":            end_sec,
                "n_frames":           len(frame_indices),
                "motion":             motion,
                "social_interaction": {
                    "label":      social,
                    "nearby_ids": nearby_ids,
                },
                "activity":           activity,
                "taxonomy":           taxonomy,
                "description":        raw,
            }
            if compare_ab is not None:
                win["compare_attn_bias"] = compare_ab
            identity_windows[ident][wid] = win

    print(f"\nReading {args.video} and running PLM on 2-sec full-frame clips …")
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        sys.exit(f"Cannot open video: {args.video}")

    current_wid: int = -1
    # Shared full-frame cache for current window (frame_idx → PIL image)
    win_frame_cache: Dict[int, PILImage.Image] = {}
    # Per-identity bbox entries for current window: [(frame_idx, cx, cy, w, h)]
    win_bbox_map: Dict[str, List] = defaultdict(list)
    frame_idx = 0

    while frame_idx < max_frames:
        ret, bgr = cap.read()
        if not ret:
            break

        wid = frame_idx // window_frames

        # Window boundary crossed — process the completed window then clear buffers
        if wid != current_wid and current_wid >= 0:
            _process_window(current_wid, win_frame_cache, win_bbox_map)
            win_frame_cache = {}
            win_bbox_map    = defaultdict(list)

        current_wid = wid

        if frame_idx in frame_map:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            # Store the full frame once, shared across all identities in this frame
            pil_frame = PILImage.fromarray(rgb)
            win_frame_cache[frame_idx] = pil_frame
            for (ident, cx, cy, w, h) in frame_map[frame_idx]:
                win_bbox_map[ident].append((frame_idx, cx, cy, w, h))

        frame_idx += 1
        if frame_idx % 60 == 0:
            print(f"\r  read {frame_idx}/{max_frames} frames", end="", flush=True)

    cap.release()
    print()

    # Process last (possibly partial) window
    if current_wid >= 0 and win_frame_cache:
        _process_window(current_wid, win_frame_cache, win_bbox_map)

    # ------------------------------------------------------------------
    # 5. Build results + write JSON
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
    # 6. Render annotated video  (second pass — no GPU work)
    # ------------------------------------------------------------------
    print(f"\nRendering overlay video …")
    frame_anns = _build_frame_annotations(tracks, max_frames)
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
            wid  = frame_idx // window_frames
            wins = identity_windows.get(ident, {})
            if wid not in wins:
                earlier = [k for k in wins if k <= wid]
                wid = max(earlier) if earlier else None  # type: ignore[assignment]
            win_data = wins.get(wid) if wid is not None else None  # type: ignore[arg-type]
            if win_data is None and frame_idx % window_frames == 0:
                print(f"  [dbg] frame {frame_idx}: no window result for {ident}")

            _draw_window_overlay(
                frame, x1, y1, x2, y2,
                ident, win_data, identity_colors[ident], args.font_scale,
            )

        # Timestamp watermark
        cv2.putText(frame, f"t={frame_idx / fps:.1f}s", (8, vid_h - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

        writer.write(frame)
        frame_idx += 1
        if frame_idx % 60 == 0:
            print(f"\r  render {frame_idx}/{max_frames} frames", end="", flush=True)

    cap.release()
    writer.release()
    print(f"\nSaved video → {out_vid}")
