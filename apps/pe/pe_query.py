"""
Interactive QA over pe_description_gen.py output JSON.

A regex router classifies each query as 'code' or 'language':
  - code     → LLM writes Python, Python executes it (exact results)
  - language → LLM reads relevant windows as text context (natural prose)

Usage
-----
    python apps/pe/pe_query.py \\
        --desc  descriptions.json \\
        --plm-ckpt facebook/Perception-LM-8B

Example queries
---------------
    > list all IDs                                        [code]
    > find who puts a toy on the shelf and at what time   [code]
    > list persons who interact with d14718               [code]
    > summarize what d14717 was doing                     [language]
    > describe the interactions between d14709 and d14718 [language]
"""

import argparse
import contextlib
import io
import json
import os
import re
from copy import deepcopy
from typing import Dict, List, Optional, Tuple

import torch


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_desc(path: str) -> Dict:
    with open(path) as f:
        return json.load(f)


def _find_mentioned_ids(text: str, known_ids) -> List[str]:
    candidates = re.findall(r'\b[a-zA-Z]\d{4,}\b', text)
    return [c for c in candidates if c in known_ids]


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

_CODE_RE = re.compile(
    r'\b(list|find|count|total|sum|accumulated|how many|how long|how much|'
    r'who|when|which|at what time|between|before|after|during|'
    r'rank|sort|compare|average|longest|shortest|most|least|'
    r'show all|show me|give me)\b',
    re.I,
)
_LANG_RE = re.compile(
    r'\b(summarize|summary|describe|explain|tell me about|'
    r'what was|what were|what did|overview|analyze|analysis|'
    r'narrative|report|elaborate|in detail)\b',
    re.I,
)

def _classify(question: str) -> str:
    """Return 'language' if the query asks for prose, else 'code'."""
    if _LANG_RE.search(question):
        return "language"
    return "code"


# ---------------------------------------------------------------------------
# PLM loader  (text-only, KV-cached)
# ---------------------------------------------------------------------------

CODEGEN_SYSTEM = (
    "You are a Python programming assistant. "
    "You will receive a description of a Python variable called `data` "
    "and a question about it. "
    "Write Python code that answers the question by operating on `data`. "
    "Rules:\n"
    "  1. Use print() to output the answer.\n"
    "  2. Output ONLY the Python code — no explanation, no markdown fences.\n"
    "  3. Do not import any modules.\n"
    "  4. Keep the code concise.\n"
    "  5. Never nest f-strings. Build complex strings in a separate variable first."
)

LANGUAGE_SYSTEM = (
    "You are an assistant analyzing person tracking and behavior data from a video. "
    "Each person has 6-second windows. Each window may contain:\n"
    "  - motion/activity/social: free-text M/S/A fields (may be empty if not collected)\n"
    "  - taxonomy: structured labels — body_state (pose), obj_verb+obj_noun (object interaction), "
    "social {label, with_ids} (social interaction with named person IDs), "
    "safety_event (safety classification)\n"
    "When M/S/A fields are empty, rely on taxonomy. "
    "Answer in clear natural language. Always cite specific timestamps."
)

DATA_SCHEMA = """\
# `data` structure
# data[person_id] = list of windows  (chronological, 6 s each)
# window keys:
#   start_sec : float        window start time (seconds)
#   end_sec   : float        window end time   (seconds)
#   motion    : str          body movement description
#   activity  : str          activity description
#   social_interaction : dict
#       label      : str        physical contact description or "none"
#       nearby_ids : list[str]  IDs of people within close proximity
#   taxonomy : dict          structured classification (separate PLM call)
#       body_state   : str   idle_stand|idle_sit|walk|walk_loaded|run|bend|squat|
#                            kneel|reach_overhead|reach_low|twist|crouch_sustained|
#                            climb|fall|recover_balance|unknown
#       obj_verb     : str   reach|grasp|lift|lower|carry|place|push|pull|drag|
#                            stack|unstack|pack|unpack|scan|inspect|operate|
#                            throw|catch|rotate|none
#       obj_noun     : str   box|pallet|scanner|cart|forklift|ladder|tool|
#                            document|shelf|bin|bag|package|none|<other>
#       social       : dict  {"label": str, "with_ids": list[str]}
#                            label    : none|talk|handover|receive|co_manipulate|
#                                       gesture_instruct|point
#                            with_ids : IDs of people involved (empty when none)
#       safety_event : str   none|zone_enter|zone_exit|ppe_don|ppe_doff|
#                            near_miss|hazard_response|fall
#       other_text   : str   free text when no slot applies (else "")
#   description : str        full raw model output

# --- EXAMPLES ---

# Q: list persons near d14718 and their accumulated time
# for pid, windows in data.items():
#     if pid == 'd14718': continue
#     near = [w for w in windows if 'd14718' in w['social_interaction']['nearby_ids']]
#     if near:
#         total = sum(w['end_sec'] - w['start_sec'] for w in near)
#         spans = []
#         for w in near:
#             spans.append(f"{w['start_sec']:.1f}s-{w['end_sec']:.1f}s")
#         print(f"{pid}: {total:.1f}s  ({', '.join(spans)})")

# Q: find who puts a toy on the shelf and when
# for pid, windows in data.items():
#     for w in windows:
#         if 'shelf' in w['activity'].lower() or 'shelf' in w['motion'].lower():
#             print(f"{pid}: {w['start_sec']:.1f}s-{w['end_sec']:.1f}s  {w['activity']}")

# Q: list body state of d14717 over time
# for w in data['d14717']:
#     tax = w.get('taxonomy', {})
#     print(f"{w['start_sec']:.1f}s: {tax.get('body_state', 'unknown')}")

# Q: find all lifting events and who did them
# for pid, windows in data.items():
#     for w in windows:
#         tax = w.get('taxonomy', {})
#         if tax.get('obj_verb') == 'lift':
#             on = tax.get('obj_noun', '')
#             print(f"{pid}: {w['start_sec']:.1f}s-{w['end_sec']:.1f}s  obj={on}")

# Q: find all co_manipulate interactions
# for pid, windows in data.items():
#     for w in windows:
#         tax = w.get('taxonomy', {})
#         sc = tax.get('social', {})
#         if isinstance(sc, dict) and sc.get('label') == 'co_manipulate':
#             with_str = ', '.join(sc.get('with_ids', []))
#             print(f"{pid}: {w['start_sec']:.1f}s  with=[{with_str}]")

# Q: list safety events
# for pid, windows in data.items():
#     for w in windows:
#         tax = w.get('taxonomy', {})
#         se = tax.get('safety_event', 'none')
#         if se != 'none':
#             print(f"{pid}: {w['start_sec']:.1f}s-{w['end_sec']:.1f}s  {se}")
"""


def _load_text_generator(ckpt: str, max_new_tokens: int):
    from apps.plm.generate import (
        PackedCausalTransformerGenerator,
        PackedCausalTransformerGeneratorArgs,
        load_consolidated_model_and_tokenizer,
    )
    from apps.plm.tokenizer import Llama3Tokenizer
    from core.data.conversation import REGISTERED_CONVS

    plm_model, _, plm_config = load_consolidated_model_and_tokenizer(ckpt)

    if os.path.exists(ckpt):
        ckpt_dir = ckpt
    else:
        from huggingface_hub import snapshot_download
        ckpt_dir = os.path.join(snapshot_download(ckpt), "original")

    tok_path = plm_config.data.tokenizer_path
    if not os.path.exists(tok_path):
        tok_path = os.path.join(ckpt_dir, tok_path)

    class _TextTok(Llama3Tokenizer):
        pass

    text_tokenizer = _TextTok(tok_path)

    rope_limit = plm_model.max_seqlen
    print(f"  model max_seqlen (RoPE limit): {rope_limit}")

    gen_cfg = PackedCausalTransformerGeneratorArgs(
        temperature=0.0,
        max_gen_len=max_new_tokens,
        max_tokens=min(rope_limit, 32768),
        dtype="bf16",
        device="cuda",
    )
    generator = PackedCausalTransformerGenerator(gen_cfg, plm_model, text_tokenizer)

    # Two conversation templates — one per mode
    code_tmpl = deepcopy(REGISTERED_CONVS["plm_sft"])
    code_tmpl.system = CODEGEN_SYSTEM

    lang_tmpl = deepcopy(REGISTERED_CONVS["plm_sft"])
    lang_tmpl.system = LANGUAGE_SYSTEM

    return generator, text_tokenizer, code_tmpl, lang_tmpl


# ---------------------------------------------------------------------------
# Code path
# ---------------------------------------------------------------------------

def _build_codegen_prompt(tmpl, ids: List[str], question: str) -> str:
    ids_line = "# Available person IDs: " + ", ".join(sorted(ids))
    full_q = f"{DATA_SCHEMA}\n{ids_line}\n\nQuestion: {question}\n\nPython code:"
    return tmpl.get_generation_prompt(full_q, num_images=0, num_patches=0)


def _extract_code(text: str) -> str:
    m = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    return text.strip()


def _execute_code(code: str, data: Dict) -> Tuple[str, Optional[str]]:
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            exec(code, {"data": data})          # noqa: S102
        return buf.getvalue().strip(), None
    except Exception as exc:
        return buf.getvalue().strip(), f"{type(exc).__name__}: {exc}"


def _ask_code(generator, tmpl, data: Dict, question: str) -> str:
    prompt = _build_codegen_prompt(tmpl, list(data.keys()), question)
    responses, _, _ = generator.generate([prompt])
    code = _extract_code(responses[0].strip())
    print(f"  [code]\n{code}\n")
    output, err = _execute_code(code, data)
    if err:
        return "\n".join(filter(None, [f"[exec error] {err}", output]))
    return output or "(no output)"


# ---------------------------------------------------------------------------
# Language path
# ---------------------------------------------------------------------------

def _format_context(data: Dict, ids: List[str], max_windows: int = 10) -> str:
    """Render the most recent max_windows per identity as plain text."""
    lines = []
    for ident in ids:
        if ident not in data:
            continue
        lines.append(f"Person {ident}:")
        for w in data[ident][-max_windows:]:
            t0, t1 = w.get("start_sec", "?"), w.get("end_sec", "?")
            motion   = w.get("motion", "")   or ""
            activity = w.get("activity", "") or ""
            si       = w.get("social_interaction", {}) or {}
            social   = (si.get("label", "") if isinstance(si, dict) else str(si)) or ""
            nearby   = si.get("nearby_ids", []) if isinstance(si, dict) else []

            # M/S/A fields (omit if empty — happens with --no-msa)
            msa_parts = []
            if motion:
                msa_parts.append(f"motion={motion!r}")
            if activity:
                msa_parts.append(f"activity={activity!r}")
            if social:
                msa_parts.append(f"social={social!r}")
            if nearby:
                msa_parts.append(f"nearby=[{', '.join(nearby)}]")

            # Taxonomy fields
            tax = w.get("taxonomy", {}) or {}
            tax_parts = []
            bs  = tax.get("body_state", "")
            ov  = tax.get("obj_verb", "none")
            on_ = tax.get("obj_noun", "none")
            sc  = tax.get("social", {})
            se  = tax.get("safety_event", "none")
            ot  = tax.get("other_text", "")
            if bs:
                tax_parts.append(f"BS={bs}")
            if ov != "none":
                tax_parts.append(f"OV={ov}")
            if on_ != "none":
                tax_parts.append(f"ON={on_}")
            if isinstance(sc, dict) and sc.get("label", "none") != "none":
                sc_str = sc["label"]
                if sc.get("with_ids"):
                    sc_str += f"[{','.join(sc['with_ids'])}]"
                tax_parts.append(f"SC={sc_str}")
            if se != "none":
                tax_parts.append(f"SE={se}")
            if ot:
                tax_parts.append(f"other={ot!r}")

            row = f"  [{t0:.1f}s-{t1:.1f}s]"
            if msa_parts:
                row += "  " + "  ".join(msa_parts)
            if tax_parts:
                row += "  taxonomy: " + "  ".join(tax_parts)
            lines.append(row)
    return "\n".join(lines)


def _ask_language(generator, tmpl, data: Dict, known_ids,
                  question: str, rope_limit: int, max_gen: int) -> str:
    # Include mentioned IDs, or all IDs if none mentioned
    mentioned = _find_mentioned_ids(question, known_ids)
    ids = mentioned if mentioned else list(data.keys())

    max_ctx_tokens = min(generator.max_tokens, rope_limit) - max_gen - 64

    # Reduce windows per identity until the prompt fits
    max_windows = 20
    while max_windows > 1:
        context = _format_context(data, ids, max_windows)
        full_q  = f"Tracking data:\n{context}\n\nQuestion: {question}"
        prompt  = tmpl.get_generation_prompt(full_q, num_images=0, num_patches=0)
        n_tok   = len(generator.tokenizer.encode(prompt, add_bos=False, add_eos=False))
        if n_tok <= max_ctx_tokens:
            break
        max_windows -= 2

    responses, _, _ = generator.generate([prompt])
    return responses[0].strip()


# ---------------------------------------------------------------------------
# Unified entry point
# ---------------------------------------------------------------------------

def _ask(generator, code_tmpl, lang_tmpl, data: Dict, known_ids,
         question: str, rope_limit: int, max_gen: int) -> str:
    mode = _classify(question)
    print(f"  [route → {mode}]")
    if mode == "code":
        return _ask_code(generator, code_tmpl, data, question)
    else:
        return _ask_language(generator, lang_tmpl, data, known_ids,
                             question, rope_limit, max_gen)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Interactive QA over descriptions.json with code/language router.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--desc", required=True, metavar="PATH")
    p.add_argument("--plm-ckpt", default="facebook/Perception-LM-8B", metavar="CKPT")
    p.add_argument("--max-new-tokens", type=int, default=512, metavar="N")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    data = _load_desc(args.desc)
    known_ids = set(data.keys())
    n_windows = sum(len(v) for v in data.values())
    print(f"Loaded {len(data)} identities, {n_windows} windows from {args.desc}")

    print(f"Loading PLM from {args.plm_ckpt} …")
    generator, _, code_tmpl, lang_tmpl = _load_text_generator(
        args.plm_ckpt, args.max_new_tokens
    )
    rope_limit = generator.model.max_seqlen

    print("Ready. Ctrl-C or Ctrl-D to quit.\n")

    while True:
        try:
            query = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not query:
            continue

        answer = _ask(generator, code_tmpl, lang_tmpl, data, known_ids,
                      query, rope_limit, args.max_new_tokens)
        print(answer)
        print()


if __name__ == "__main__":
    main()
