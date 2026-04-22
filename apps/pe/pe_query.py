"""
Interactive QA over pe_description_gen.py output JSON using PLM's LLM.

The descriptions JSON is formatted as text context and fed to the PLM
language model (text-only, no images), using the built-in KV cache for
efficient generation.

Usage
-----
    python apps/pe/pe_query.py \\
        --desc  descriptions.json \\
        --plm-ckpt facebook/Perception-LM-8B

    # Limit context to specific identities:
    python apps/pe/pe_query.py \\
        --desc descriptions.json \\
        --plm-ckpt facebook/Perception-LM-8B \\
        --ids d14717 d14709

Example queries
---------------
    > What is d14717 doing?
    > When does d14709 interact with someone?
    > What is d14717 doing at 36 seconds?
    > Which two people are closest together?
    > Is anyone picking something up?
"""

import argparse
import json
import os
import re
from copy import deepcopy
from typing import Dict, List, Optional

import torch


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _load_desc(path: str) -> Dict:
    with open(path) as f:
        return json.load(f)


def _find_mentioned_ids(text: str, known_ids) -> List[str]:
    candidates = re.findall(r'\b[a-zA-Z]\d{4,}\b', text)
    return [c for c in candidates if c in known_ids]


def _format_context(data: Dict, ids: Optional[List[str]] = None) -> str:
    """Render the JSON as compact structured text for LLM context."""
    ids = ids or list(data.keys())
    lines = []
    for ident in ids:
        if ident not in data:
            continue
        lines.append(f"Person {ident}:")
        for w in data[ident]:
            t0, t1 = w.get("start_sec", "?"), w.get("end_sec", "?")
            motion   = w.get("motion",   "") or "unclear"
            activity = w.get("activity", "") or "unclear"
            si       = w.get("social_interaction", {}) or {}
            social   = (si.get("label", "") if isinstance(si, dict) else str(si)) or "none"
            nearby   = si.get("nearby_ids", []) if isinstance(si, dict) else []
            nb_str   = f"  nearby=[{', '.join(nearby)}]" if nearby else ""
            lines.append(
                f"  [{t0:.1f}s-{t1:.1f}s] "
                f"motion={motion!r}  activity={activity!r}  social={social!r}{nb_str}"
            )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# PLM text-only generation with KV cache
# ---------------------------------------------------------------------------

SYSTEM = (
    "You are an assistant analyzing person tracking and behavior data extracted "
    "from a video. Each person has a list of 6-second windows with fields: "
    "motion (body movement), activity (what they are doing), social (physical "
    "contact with others), and nearby (people within close proximity). "
    "When answering, always include the specific time window (e.g. 36.0s-42.0s). "
    "Be concise and factual."
)


def _load_text_generator(ckpt: str, max_new_tokens: int):
    """
    Load PLM and return a KV-cached generator for text-only inference.

    PackedCausalTransformerGenerator.generate() takes text-only prompts
    (plain strings, not (prompt, image) tuples) when the tokenizer is NOT
    a PLMTokenizer.  We create a Llama3Tokenizer subclass that passes the
    isinstance check correctly, then wire it up to the cached generator.
    """
    from apps.plm.generate import (
        PackedCausalTransformerGenerator,
        PackedCausalTransformerGeneratorArgs,
        load_consolidated_model_and_tokenizer,
    )
    from apps.plm.tokenizer import Llama3Tokenizer
    from core.data.conversation import REGISTERED_CONVS

    plm_model, _, plm_config = load_consolidated_model_and_tokenizer(ckpt)

    # Resolve tokenizer file path (mirrors load_consolidated_model_and_tokenizer logic)
    if os.path.exists(ckpt):
        ckpt_dir = ckpt
    else:
        from huggingface_hub import snapshot_download
        ckpt_dir = os.path.join(snapshot_download(ckpt), "original")

    tok_path = plm_config.data.tokenizer_path
    if not os.path.exists(tok_path):
        tok_path = os.path.join(ckpt_dir, tok_path)

    # Subclass Llama3Tokenizer so isinstance(tok, PLMTokenizer) == False,
    # which routes generate() through the text-only (KV-cached) code path.
    class _TextTok(Llama3Tokenizer):
        pass

    text_tokenizer = _TextTok(tok_path)

    # max_tokens must not exceed the model's RoPE limit (max_seqlen).
    # freq_cis is precomputed only up to max_seqlen; exceeding it causes
    # a shape mismatch in apply_rotary_emb that explodes silently on GPU.
    rope_limit = plm_model.max_seqlen
    effective_max = min(rope_limit, 32768)
    print(f"  model max_seqlen (RoPE limit): {rope_limit}")

    gen_cfg = PackedCausalTransformerGeneratorArgs(
        temperature=0.0,
        max_gen_len=max_new_tokens,
        max_tokens=effective_max,
        dtype="bf16",
        device="cuda",
    )
    generator = PackedCausalTransformerGenerator(gen_cfg, plm_model, text_tokenizer)

    conv_template = deepcopy(REGISTERED_CONVS["plm_sft"])
    conv_template.system = SYSTEM

    return generator, text_tokenizer, conv_template


def _keyword_filter(data: Dict, query: str, top_k: int = 30) -> Dict:
    """
    Score every window by keyword overlap with the query and return the top_k
    highest-scoring windows grouped by identity.

    This runs before the LLM so:
      - Only relevant windows reach the context (no trimming of key evidence)
      - Unrelated identities are excluded, preventing identity confusion
    """
    STOP = {
        "the", "a", "an", "is", "was", "who", "what", "when", "where",
        "find", "show", "tell", "me", "and", "or", "of", "to", "in",
        "at", "for", "its", "their", "his", "her", "he", "she", "they",
        "did", "does", "do", "has", "have", "had", "with",
    }
    keywords = [
        w.lower() for w in re.findall(r"\b\w+\b", query)
        if w.lower() not in STOP and len(w) > 2
    ]
    if not keywords:
        return data   # no keywords → return everything

    scored: List[tuple] = []
    for ident, windows in data.items():
        for win in windows:
            text = " ".join(filter(None, [
                win.get("motion", ""),
                win.get("activity", ""),
                win.get("description", ""),
            ])).lower()
            score = sum(1 for kw in keywords if kw in text)
            if score > 0:
                scored.append((score, ident, win))

    if not scored:
        return data   # no matches → fall back to full context

    scored.sort(key=lambda x: -x[0])

    filtered: Dict[str, List] = {}
    for _, ident, win in scored[:top_k]:
        filtered.setdefault(ident, []).append(win)

    print(f"  [filter] {len(scored)} matching windows → top {len(filtered)} identities")
    return filtered


def _compute_interaction_stats(data: Dict, target_id: str) -> str:
    """
    For every person X in data, compute:
      - windows where target_id appears in X's nearby_ids
      - accumulated duration of those windows

    Returns a pre-computed summary string to inject into the prompt so the
    LLM never has to do arithmetic (which it reliably gets wrong).
    """
    rows = []
    for ident, windows in data.items():
        if ident == target_id:
            continue
        matching = [
            w for w in windows
            if target_id in (
                (w.get("social_interaction") or {}).get("nearby_ids", [])
            )
        ]
        if not matching:
            continue
        total_sec = sum(w.get("end_sec", 0) - w.get("start_sec", 0) for w in matching)
        spans = ", ".join(f"{w['start_sec']:.1f}s-{w['end_sec']:.1f}s" for w in matching)
        rows.append(f"  {ident}: {total_sec:.1f}s total  ({spans})")
    if not rows:
        return f"Pre-computed: no one has {target_id} in their nearby_ids.\n"
    return "Pre-computed proximity stats with " + target_id + ":\n" + "\n".join(rows) + "\n"


def _build_prompt(conv_template, context: str, question: str,
                  stats: str = "") -> str:
    full_q = f"{stats}Tracking data:\n{context}\n\nQuestion: {question}"
    return conv_template.get_generation_prompt(full_q, num_images=0, num_patches=0)


def _ask(generator, conv_template, data: Dict, context_ids: List[str],
         question: str) -> str:
    # Step 1: Pre-compute interaction stats if a specific ID is mentioned.
    mentioned = _find_mentioned_ids(question, set(data.keys()))
    stats = ""
    if mentioned and re.search(r"\binteract|accum|total time|how long|how much time\b",
                               question, re.I):
        for mid in mentioned:
            stats += _compute_interaction_stats(data, mid)

    # Step 2: keyword pre-filter so only relevant windows reach the LLM.
    scoped = {k: v for k, v in data.items() if k in context_ids}
    filtered = _keyword_filter(scoped, question)

    # Step 3: Hard limit is the smaller of KV cache size and RoPE table size.
    rope_limit = generator.model.max_seqlen
    max_prompt_tokens = (min(generator.max_tokens, rope_limit)
                         - generator.max_gen_len - 64
                         - len(generator.tokenizer.encode(stats, add_bos=False, add_eos=False)))

    # Step 4: Trim if still too long (drop oldest windows from largest identity).
    recent_windows = {ident: len(wins) for ident, wins in filtered.items()}
    ids_ordered = list(filtered.keys())
    while True:
        context = _format_context_windowed(filtered, ids_ordered, recent_windows)
        prompt  = _build_prompt(conv_template, context, question, stats)
        n_tok   = len(generator.tokenizer.encode(prompt, add_bos=False, add_eos=False))
        if n_tok <= max_prompt_tokens:
            break
        largest = max(recent_windows, key=lambda k: recent_windows[k])
        recent_windows[largest] -= 1
        if all(v <= 1 for v in recent_windows.values()):
            break

    responses, _, _ = generator.generate([prompt])
    return responses[0].strip()


def _format_context_windowed(
    data: Dict, ids: List[str], recent_windows: Dict[str, int]
) -> str:
    """Like _format_context but only shows the last N windows per identity."""
    lines = []
    for ident in ids:
        if ident not in data:
            continue
        windows = data[ident][-recent_windows.get(ident, len(data[ident])):]
        lines.append(f"Person {ident}:")
        for w in windows:
            t0, t1 = w.get("start_sec", "?"), w.get("end_sec", "?")
            motion   = w.get("motion",   "") or "unclear"
            activity = w.get("activity", "") or "unclear"
            si       = w.get("social_interaction", {}) or {}
            social   = (si.get("label", "") if isinstance(si, dict) else str(si)) or "none"
            nearby   = si.get("nearby_ids", []) if isinstance(si, dict) else []
            nb_str   = f"  nearby=[{', '.join(nearby)}]" if nearby else ""
            lines.append(
                f"  [{t0:.1f}s-{t1:.1f}s] "
                f"motion={motion!r}  activity={activity!r}  social={social!r}{nb_str}"
            )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Interactive QA over descriptions.json using PLM LLM.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--desc", required=True, metavar="PATH",
                   help="descriptions.json from pe_description_gen.py")
    p.add_argument("--plm-ckpt", default="facebook/Perception-LM-8B", metavar="CKPT",
                   help="PLM checkpoint or HF model ID (default: facebook/Perception-LM-8B)")
    p.add_argument("--ids", nargs="*", metavar="ID",
                   help="Limit context to these identity IDs (default: all)")
    p.add_argument("--max-new-tokens", type=int, default=512, metavar="N",
                   help="Max tokens to generate per answer (default: 512)")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    data = _load_desc(args.desc)
    known_ids = set(data.keys())
    n_windows = sum(len(v) for v in data.values())
    print(f"Loaded {len(data)} identities, {n_windows} windows from {args.desc}")

    print(f"Loading PLM from {args.plm_ckpt} …")
    generator, _, conv_template = _load_text_generator(args.plm_ckpt, args.max_new_tokens)

    print("Ready. Ask a question about any tracked person. Ctrl-C or Ctrl-D to quit.\n")

    while True:
        try:
            query = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not query:
            continue

        mentioned = _find_mentioned_ids(query, known_ids)
        context_ids = mentioned if mentioned else (args.ids or list(data.keys()))

        answer = _ask(generator, conv_template, data, context_ids, query)
        print(answer)
        print()


if __name__ == "__main__":
    main()
