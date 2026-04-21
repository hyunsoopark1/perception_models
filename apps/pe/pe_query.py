"""
Interactive QA over pe_description_gen.py output JSON using PLM's LLM.

The full descriptions JSON is formatted as text context and fed to the
PLM language model (text-only, no images).  The LLM answers natural-language
questions about any tracked identity, always citing specific timestamps.

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
import re
import sys
from copy import deepcopy
from pathlib import Path
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
# PLM text-only generation
# ---------------------------------------------------------------------------

SYSTEM = (
    "You are an assistant analyzing person tracking and behavior data extracted "
    "from a video. Each person has a list of 6-second windows with fields: "
    "motion (body movement), activity (what they are doing), social (physical "
    "contact with others), and nearby (people within close proximity). "
    "When answering, always include the specific time window (e.g. 36.0s-42.0s). "
    "Be concise and factual."
)


def _build_prompt(conv_template, context: str, question: str) -> str:
    full_q = f"Tracking data:\n{context}\n\nQuestion: {question}"
    # Use text-only mode: num_images=0, num_patches=0
    return conv_template.get_generation_prompt(full_q, num_images=0, num_patches=0)


@torch.inference_mode()
def _generate(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 300,
) -> str:
    token_ids = tokenizer.encode(prompt, add_bos=False, add_eos=False)
    input_ids = torch.tensor([token_ids], dtype=torch.long).cuda()
    generated: List[int] = []

    for _ in range(max_new_tokens):
        logits = model(input_ids, attn_impl="sdpa")   # (1, seqlen, vocab)
        next_tok = int(logits[0, -1, :].argmax())
        if next_tok in (tokenizer.eos_id, tokenizer.eot_id):
            break
        generated.append(next_tok)
        next_tensor = torch.tensor([[next_tok]], dtype=torch.long, device=input_ids.device)
        input_ids = torch.cat([input_ids, next_tensor], dim=-1)

    return tokenizer.decode(generated).strip()


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
    p.add_argument("--max-new-tokens", type=int, default=300, metavar="N",
                   help="Maximum tokens to generate per answer (default: 300)")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    # Load descriptions
    data = _load_desc(args.desc)
    known_ids = set(data.keys())
    n_windows = sum(len(v) for v in data.values())
    print(f"Loaded {len(data)} identities, {n_windows} windows from {args.desc}")

    # Load PLM
    print(f"Loading PLM from {args.plm_ckpt} …")
    from apps.plm.generate import (
        load_consolidated_model_and_tokenizer,
    )
    from core.data.conversation import REGISTERED_CONVS

    plm_model, plm_tokenizer, _ = load_consolidated_model_and_tokenizer(args.plm_ckpt)

    # Override the system message with our domain-specific one
    conv_template = deepcopy(REGISTERED_CONVS["plm_sft"])
    conv_template.system = SYSTEM

    print("Ready. Ask a question about any tracked person. Ctrl-C or Ctrl-D to quit.\n")

    while True:
        try:
            query = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not query:
            continue

        # Decide which IDs to include in context
        mentioned = _find_mentioned_ids(query, known_ids)
        context_ids = mentioned if mentioned else (args.ids or list(data.keys()))
        context = _format_context(data, context_ids)

        prompt = _build_prompt(conv_template, context, query)
        answer = _generate(plm_model, plm_tokenizer, prompt, args.max_new_tokens)
        print(answer)
        print()


if __name__ == "__main__":
    main()
