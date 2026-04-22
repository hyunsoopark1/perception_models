"""
Interactive QA over pe_description_gen.py output JSON.

The PLM LLM is used as a code generator: given the data schema and the
user's question it writes Python code, which is then executed against the
real `data` dict.  All arithmetic, aggregation, and lookups are done by
the executed code — not by the LLM — so results are always exact.

Usage
-----
    python apps/pe/pe_query.py \\
        --desc  descriptions.json \\
        --plm-ckpt facebook/Perception-LM-8B

Example queries
---------------
    > list all IDs
    > find who puts a toy on the shelf and at what time
    > list persons who interact with d14718 and their accumulated time
    > who is near d14709 at 36 seconds?
    > which person has the longest total activity duration?
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


# ---------------------------------------------------------------------------
# PLM loader  (text-only, KV-cached)
# ---------------------------------------------------------------------------

# System prompt: instruct LLM to produce only executable Python
CODEGEN_SYSTEM = (
    "You are a Python programming assistant. "
    "You will receive a description of a Python variable called `data` "
    "and a question about it. "
    "Write Python code that answers the question by operating on `data`. "
    "Rules:\n"
    "  1. Use print() to output the answer.\n"
    "  2. Output ONLY the Python code — no explanation, no markdown fences.\n"
    "  3. Do not import any modules.\n"
    "  4. Keep the code concise."
)

# Schema shown to the LLM every call (short enough to fit in context)
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
#   description : str        full raw model output

# --- EXAMPLES ---

# Q: list persons near d14718 and their accumulated time
# for pid, windows in data.items():
#     if pid == 'd14718': continue
#     near = [w for w in windows if 'd14718' in w['social_interaction']['nearby_ids']]
#     if near:
#         total = sum(w['end_sec'] - w['start_sec'] for w in near)
#         spans = ', '.join(f"{w['start_sec']:.1f}s-{w['end_sec']:.1f}s" for w in near)
#         print(f"{pid}: {total:.1f}s  ({spans})")

# Q: find who puts a toy on the shelf and when
# for pid, windows in data.items():
#     for w in windows:
#         if 'shelf' in w['activity'].lower() or 'shelf' in w['motion'].lower():
#             print(f"{pid}: {w['start_sec']:.1f}s-{w['end_sec']:.1f}s  {w['activity']}")
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

    conv_template = deepcopy(REGISTERED_CONVS["plm_sft"])
    conv_template.system = CODEGEN_SYSTEM

    return generator, conv_template


# ---------------------------------------------------------------------------
# Code generation + execution
# ---------------------------------------------------------------------------

def _build_codegen_prompt(conv_template, ids: List[str], question: str) -> str:
    ids_line = "# Available person IDs: " + ", ".join(sorted(ids))
    full_q = f"{DATA_SCHEMA}\n{ids_line}\n\nQuestion: {question}\n\nPython code:"
    return conv_template.get_generation_prompt(full_q, num_images=0, num_patches=0)


def _extract_code(text: str) -> str:
    """Pull code out of the LLM response, stripping markdown fences if present."""
    # ```python ... ``` or ``` ... ```
    m = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    # No fences — treat the whole response as code
    return text.strip()


def _execute_code(code: str, data: Dict) -> Tuple[str, Optional[str]]:
    """
    Execute *code* with `data` in scope.
    Returns (stdout, error_message).  error_message is None on success.
    """
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            exec(code, {"data": data})          # noqa: S102
        return buf.getvalue().strip(), None
    except Exception as exc:
        return buf.getvalue().strip(), f"{type(exc).__name__}: {exc}"


def _ask(generator, conv_template, data: Dict, question: str) -> str:
    ids = list(data.keys())
    prompt = _build_codegen_prompt(conv_template, ids, question)

    responses, _, _ = generator.generate([prompt])
    raw = responses[0].strip()
    code = _extract_code(raw)

    print(f"  [code]\n{code}\n")

    output, err = _execute_code(code, data)

    if err:
        # On error show the problem so the user can diagnose
        lines = [f"[exec error] {err}"]
        if output:
            lines.append(output)
        return "\n".join(lines)

    return output or "(no output)"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Interactive QA over descriptions.json — LLM writes code, Python runs it.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--desc", required=True, metavar="PATH",
                   help="descriptions.json from pe_description_gen.py")
    p.add_argument("--plm-ckpt", default="facebook/Perception-LM-8B", metavar="CKPT",
                   help="PLM checkpoint or HF model ID (default: facebook/Perception-LM-8B)")
    p.add_argument("--max-new-tokens", type=int, default=512, metavar="N",
                   help="Max tokens the LLM may generate per code snippet (default: 512)")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    data = _load_desc(args.desc)
    n_windows = sum(len(v) for v in data.values())
    print(f"Loaded {len(data)} identities, {n_windows} windows from {args.desc}")

    print(f"Loading PLM from {args.plm_ckpt} …")
    generator, conv_template = _load_text_generator(args.plm_ckpt, args.max_new_tokens)

    print("Ready. Ask a question about the tracked persons. Ctrl-C or Ctrl-D to quit.\n")

    while True:
        try:
            query = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not query:
            continue

        answer = _ask(generator, conv_template, data, query)
        print(answer)
        print()


if __name__ == "__main__":
    main()
