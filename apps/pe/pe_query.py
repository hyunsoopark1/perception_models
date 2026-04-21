"""
Interactive query interface over pe_description_gen.py output JSON.

Parses natural-language questions, extracts referenced identity IDs, and
returns answers with specific timestamps.

Usage
-----
    python apps/pe/pe_query.py --desc descriptions.json

Example queries
---------------
    > What is d14717 doing?
    > When does d14709 interact with someone?
    > What is d14717 doing at 36 seconds?
    > Who is near d14709?
    > Show all windows for d14715
    > When does d14718 pick something up?
"""

import argparse
import json
import re
import sys
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load(path: str) -> Dict:
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Query parsing helpers
# ---------------------------------------------------------------------------

def _find_ids(text: str, known_ids: set) -> List[str]:
    """Return all known identity IDs mentioned in *text*."""
    candidates = re.findall(r'\b[a-zA-Z]\d{4,}\b', text)
    return [c for c in candidates if c in known_ids]


def _find_time_ref(text: str) -> Optional[float]:
    """
    Extract a time reference in seconds from text, e.g.:
      'at 36 seconds', 'at 36s', 'at t=36', 'around 1:30'
    """
    m = re.search(r'\bat\s+(?:t\s*=\s*)?(\d+(?:\.\d+)?)\s*s(?:ec(?:onds?)?)?\b', text, re.I)
    if m:
        return float(m.group(1))
    # mm:ss format
    m = re.search(r'\b(\d+):(\d{2})\b', text)
    if m:
        return int(m.group(1)) * 60 + int(m.group(2))
    return None


def _question_type(text: str) -> str:
    """
    Classify the question intent:
      'social'   — asking about interactions / nearby people
      'motion'   — asking about movement
      'activity' — asking about what they're doing
      'when'     — asking for time of a specific event
      'all'      — general / show everything
    """
    low = text.lower()
    if re.search(r'\binteract|social|touch|near|with whom|who.*near|next to\b', low):
        return 'social'
    if re.search(r'\bmov|walk|run|stand|sit|motion\b', low):
        return 'motion'
    if re.search(r'\bwhen\b', low):
        return 'when'
    if re.search(r'\bdoing|activity|action|what\b', low):
        return 'activity'
    return 'all'


# ---------------------------------------------------------------------------
# Answer formatters
# ---------------------------------------------------------------------------

def _fmt_time(start: float, end: float) -> str:
    return f"[{start:.1f}s – {end:.1f}s]"


def _fmt_window(win: Dict, fields: str = 'all') -> str:
    t = _fmt_time(win['start_sec'], win['end_sec'])
    lines = [f"  {t}"]

    motion   = win.get('motion', '') or '—'
    activity = win.get('activity', '') or '—'
    sdict    = win.get('social_interaction', {}) or {}
    social   = (sdict.get('label', '') if isinstance(sdict, dict) else str(sdict)) or 'none'
    nearby   = sdict.get('nearby_ids', []) if isinstance(sdict, dict) else []

    if fields in ('all', 'motion'):
        lines.append(f"    Motion:   {motion}")
    if fields in ('all', 'activity'):
        lines.append(f"    Activity: {activity}")
    if fields in ('all', 'social'):
        lines.append(f"    Social:   {social}")
        if nearby:
            lines.append(f"    Nearby:   {', '.join(nearby)}")

    return '\n'.join(lines)


def _answer_identity(ident: str, windows: List[Dict], qtype: str, t_ref: Optional[float]) -> str:
    if not windows:
        return f"{ident}: no data."

    # Filter to the window containing t_ref if specified
    if t_ref is not None:
        matched = [w for w in windows if w['start_sec'] <= t_ref < w['end_sec']]
        if not matched:
            # Snap to closest window
            matched = [min(windows, key=lambda w: abs(w['start_sec'] - t_ref))]
            snap_t  = matched[0]['start_sec']
            header  = f"{ident} (nearest window to t={t_ref:.1f}s, starts at {snap_t:.1f}s):"
        else:
            header = f"{ident} at t={t_ref:.1f}s:"
        windows = matched

    # Filter by question type
    elif qtype == 'social':
        windows = [
            w for w in windows
            if (w.get('social_interaction', {}) or {}).get('label', '').lower() not in ('', 'none')
        ]
        if not windows:
            return f"{ident}: no social interactions detected."
        header = f"{ident} — social interactions:"

    elif qtype == 'when':
        # Return all windows with a non-trivial activity field
        windows = [w for w in windows if w.get('activity', '')]
        header  = f"{ident} — activity timeline:"

    else:
        header = f"{ident}:"

    field_map = {'motion': 'motion', 'activity': 'activity', 'social': 'social'}
    fields = field_map.get(qtype, 'all')

    summaries = '\n'.join(_fmt_window(w, fields) for w in windows)
    return f"{header}\n{summaries}"


# ---------------------------------------------------------------------------
# Top-level answer function
# ---------------------------------------------------------------------------

def answer(query: str, data: Dict) -> str:
    known_ids = set(data.keys())
    mentioned = _find_ids(query, known_ids)
    t_ref     = _find_time_ref(query)
    qtype     = _question_type(query)

    if not mentioned:
        ids_hint = ', '.join(sorted(known_ids))
        return f"No known identity found in query.\nKnown IDs: {ids_hint}"

    parts = [
        _answer_identity(ident, data[ident], qtype, t_ref)
        for ident in mentioned
    ]
    return '\n\n'.join(parts)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description="Interactive query over pe_description_gen output JSON.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--desc", required=True, metavar="PATH",
                   help="Path to descriptions.json produced by pe_description_gen.py")
    args = p.parse_args()

    data = _load(args.desc)
    n_windows = sum(len(v) for v in data.values())
    print(f"Loaded {len(data)} identities, {n_windows} windows from {args.desc}")
    print("Ask a question referencing an ID (e.g. 'd14717'). Ctrl-C or Ctrl-D to quit.\n")

    while True:
        try:
            query = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not query:
            continue
        print(answer(query, data))
        print()


if __name__ == "__main__":
    main()
