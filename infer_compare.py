#!/usr/bin/env python3
"""
Inference + comparison script for fine-tuned vs pre-trained VLM.

Usage
-----
# Compare base model vs a LoRA checkpoint on a folder of images:
python infer_compare.py \
    --image_dir ~/data/storyline_mm/0000042 \
    --finetuned_ckpt runs/storyline_mm/checkpoint-200 \
    --base_model meta-llama/Llama-3.2-11B-Vision-Instruct

# Skip the base-model comparison (only run fine-tuned):
python infer_compare.py \
    --image_dir ~/data/storyline_mm/0000042 \
    --finetuned_ckpt runs/storyline_mm/checkpoint-200 \
    --skip_base

# Custom prompt:
python infer_compare.py \
    --image_dir ~/data/storyline_mm/0000042 \
    --finetuned_ckpt runs/storyline_mm/checkpoint-200 \
    --prompt "Describe what is happening in these images."
"""

import argparse
import textwrap
from pathlib import Path
from typing import List, Optional

import torch
from PIL import Image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

DEFAULT_PROMPT = "Look at the image(s) and write a descriptive story."
DEFAULT_BASE_MODEL = "meta-llama/Llama-3.2-11B-Vision-Instruct"


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _load_processor(model_name_or_path: str, hf_token: Optional[str] = None):
    from transformers import AutoProcessor  # type: ignore
    processor = AutoProcessor.from_pretrained(
        model_name_or_path, token=hf_token, trust_remote_code=True
    )
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    return processor


def _load_base_model(model_name: str, torch_dtype, hf_token: Optional[str] = None):
    """Load the unmodified pre-trained model."""
    import transformers as _tr  # type: ignore
    from transformers import AutoConfig  # type: ignore

    cfg = AutoConfig.from_pretrained(model_name, token=hf_token, trust_remote_code=True)
    arch = (cfg.architectures or [""])[0]

    if hasattr(_tr, arch):
        model_cls = getattr(_tr, arch)
    else:
        try:
            from transformers import AutoModelForVision2Seq  # type: ignore
            model_cls = AutoModelForVision2Seq
        except ImportError:
            from transformers import AutoModel  # type: ignore
            model_cls = AutoModel

    model = model_cls.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map="auto",
        token=hf_token,
        trust_remote_code=True,
    )
    model.eval()
    return model


def _load_finetuned_model(ckpt_path: str, torch_dtype, hf_token: Optional[str] = None):
    """
    Load a LoRA fine-tuned checkpoint saved by finetune_storyline_mm.py.

    The checkpoint may be:
      - A PEFT LoRA adapter directory  (adapter_config.json present)
      - A merged full-weight directory (no adapter_config.json)
    """
    from pathlib import Path as _P
    from peft import PeftModel  # type: ignore

    ckpt = _P(ckpt_path)

    # Read base model name from adapter_config if available
    adapter_cfg = ckpt / "adapter_config.json"
    if adapter_cfg.exists():
        import json
        cfg = json.loads(adapter_cfg.read_text())
        base_model_name = cfg.get("base_model_name_or_path", DEFAULT_BASE_MODEL)
        print(f"[info] LoRA adapter detected. Base model: {base_model_name}")
        base = _load_base_model(base_model_name, torch_dtype, hf_token)
        model = PeftModel.from_pretrained(base, str(ckpt))
    else:
        # Treat as a merged / full-weight checkpoint
        print("[info] No adapter_config.json found — loading as full-weight checkpoint.")
        model = _load_base_model(str(ckpt), torch_dtype, hf_token)

    model.eval()
    return model


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def build_inputs(processor, images: List[Image.Image], prompt: str):
    """Build model inputs from a list of PIL images and a text prompt."""
    user_content = [{"type": "image"} for _ in images]
    user_content.append({"type": "text", "text": prompt})

    messages = [{"role": "user", "content": user_content}]
    text = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )

    enc = processor(text=text, images=images, return_tensors="pt")

    # Normalise pixel_values: (1, N, C, H, W) → (N, C, H, W) when needed
    pv = enc.get("pixel_values")
    if pv is not None and pv.ndim == 5:
        enc["pixel_values"] = pv[0]

    return enc


@torch.inference_mode()
def generate(
    model,
    inputs,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> str:
    device = next(model.parameters()).device
    model_inputs = {k: v.to(device) for k, v in inputs.items()}

    output_ids = model.generate(
        **model_inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=temperature > 0,
    )

    # Strip the prompt tokens; keep only generated tokens
    prompt_len = model_inputs["input_ids"].shape[-1]
    generated = output_ids[0][prompt_len:]

    # Decode — use the processor's tokenizer if available
    tokenizer = getattr(model, "processor", None)
    if tokenizer is None:
        # Retrieve via peft base_model if wrapped
        tokenizer = getattr(getattr(model, "base_model", model), "processor", None)
    # Fall back: we'll pass tokenizer separately (see run())
    return generated


# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------

def _wrap(text: str, width: int = 90) -> str:
    return "\n".join(
        textwrap.fill(line, width=width) if line.strip() else ""
        for line in text.splitlines()
    )


def _banner(title: str, char: str = "=", width: int = 92):
    pad = max(0, (width - len(title) - 2) // 2)
    return f"\n{char * pad} {title} {char * pad}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def collect_images(image_dir: str) -> List[Path]:
    """Return image paths from a directory.

    Tries the canonical image_0 … image_14 naming convention first
    (matches the dataset layout from finetune_storyline_mm.py).
    Falls back to all image files in the directory sorted by name.
    """
    d = Path(image_dir).expanduser()
    if not d.is_dir():
        raise NotADirectoryError(f"Not a directory: {d}")

    paths: List[Path] = []
    for idx in range(5):
        for ext in IMAGE_EXTENSIONS:
            p = d / f"image_{idx}{ext}"
            if p.exists():
                paths.append(p)
                break

    if not paths:
        paths = sorted(p for p in d.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)[:5]

    if not paths:
        raise FileNotFoundError(f"No image files found in: {d}")
    return paths


def run(args):
    torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    hf_token = args.hf_token or None

    # ---- collect images ----
    image_paths = collect_images(args.image_dir)
    images = [Image.open(p).convert("RGB") for p in image_paths]
    print(f"[info] Loaded {len(images)} image(s): {[p.name for p in image_paths]}")

    # ---- load fine-tuned model ----
    print(f"\n[info] Loading fine-tuned model from: {args.finetuned_ckpt}")
    ft_model = _load_finetuned_model(args.finetuned_ckpt, torch_dtype, hf_token)

    # Processor: prefer checkpoint's own saved processor, fall back to base
    ft_proc_path = args.finetuned_ckpt
    try:
        ft_processor = _load_processor(ft_proc_path, hf_token)
    except Exception:
        ft_processor = _load_processor(DEFAULT_BASE_MODEL, hf_token)

    # ---- build inputs for fine-tuned ----
    ft_inputs = build_inputs(ft_processor, images, args.prompt)
    raw_ft = generate(ft_model, ft_inputs, args.max_new_tokens, args.temperature, args.top_p)
    ft_text = ft_processor.tokenizer.decode(raw_ft, skip_special_tokens=True)

    # Free fine-tuned model to save VRAM before loading base
    if not args.skip_base:
        del ft_model
        torch.cuda.empty_cache()

    # ---- load and run base model ----
    base_text = None
    if not args.skip_base:
        base_model_name = args.base_model
        # Try to read base model name from adapter_config
        import json
        adapter_cfg = Path(args.finetuned_ckpt) / "adapter_config.json"
        if adapter_cfg.exists():
            cfg = json.loads(adapter_cfg.read_text())
            base_model_name = cfg.get("base_model_name_or_path", base_model_name)

        print(f"\n[info] Loading base model: {base_model_name}")
        base_model = _load_base_model(base_model_name, torch_dtype, hf_token)
        try:
            base_processor = _load_processor(base_model_name, hf_token)
        except Exception:
            base_processor = ft_processor

        base_inputs = build_inputs(base_processor, images, args.prompt)
        raw_base = generate(base_model, base_inputs, args.max_new_tokens, args.temperature, args.top_p)
        base_text = base_processor.tokenizer.decode(raw_base, skip_special_tokens=True)

    # ---- print results ----
    print(_banner("INPUT"))
    print(f"  Prompt  : {args.prompt}")
    print(f"  Images  : {[p.name for p in image_paths]}")

    print(_banner("FINE-TUNED OUTPUT"))
    print(_wrap(ft_text))

    if not args.skip_base:
        print(_banner("BASE MODEL OUTPUT"))
        print(_wrap(base_text))

        print(_banner("DIFF SUMMARY"))
        ft_words = set(ft_text.lower().split())
        base_words = set(base_text.lower().split())
        unique_ft = ft_words - base_words
        unique_base = base_words - ft_words
        overlap = ft_words & base_words
        print(f"  Fine-tuned words  : {len(ft_words)}")
        print(f"  Base model words  : {len(base_words)}")
        print(f"  Shared vocabulary : {len(overlap)}")
        print(f"  Unique to FT      : {len(unique_ft)}")
        print(f"  Unique to base    : {len(unique_base)}")

    print("\n" + "=" * 92 + "\n")


def parse_args():
    p = argparse.ArgumentParser(
        description="Run inference with fine-tuned and base VLM; compare outputs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Images
    p.add_argument(
        "--image_dir", required=True, metavar="DIR",
        help="Directory containing image_0.jpg … or arbitrary image files.",
    )

    # Models
    p.add_argument(
        "--finetuned_ckpt", required=True,
        help="Path to the fine-tuned checkpoint directory (LoRA adapter or merged).",
    )
    p.add_argument(
        "--base_model", default=DEFAULT_BASE_MODEL,
        help=f"HuggingFace model ID for the base model (default: {DEFAULT_BASE_MODEL}).",
    )
    p.add_argument("--skip_base", action="store_true",
                   help="Skip loading the base model (only run fine-tuned).")
    p.add_argument("--hf_token", default=None, help="HuggingFace API token.")

    # Prompt
    p.add_argument("--prompt", default=DEFAULT_PROMPT,
                   help="Instruction prompt sent to both models.")

    # Generation
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_p", type=float, default=0.9)

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args)
