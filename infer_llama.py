#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
Text-completion inference for a (fine-tuned) Llama / TinyLlama model.

Modes
-----
1. Single model  — omit --lora_dir
2. Side-by-side comparison  — pass --lora_dir to compare base vs fine-tuned
3. Interactive REPL  — omit --prompt in either mode

Usage examples
--------------
# Base model only:
python infer_llama.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --prompt "오늘 날씨가"

# Compare base vs fine-tuned (side-by-side):
python infer_llama.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --lora_dir runs/llama_korean/checkpoint-500 \
    --prompt "서울은"

# Interactive comparison REPL:
python infer_llama.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --lora_dir runs/llama_korean/checkpoint-500
"""

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_base_model(model_name: str, dtype: str):
    torch_dtype = getattr(torch, dtype)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer


def load_finetuned_model(model_name: str, lora_dir: str, dtype: str):
    """Load base weights then merge LoRA adapters on top."""
    torch_dtype = getattr(torch, dtype)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    from peft import PeftModel  # type: ignore
    model = PeftModel.from_pretrained(model, lora_dir)
    model = model.merge_and_unload()
    model.eval()
    return model, tokenizer


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate(model, tokenizer, prompt: str, **kwargs) -> str:
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_len = inputs["input_ids"].shape[-1]
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            do_sample=kwargs["temperature"] > 0.0,
            temperature=kwargs["temperature"] if kwargs["temperature"] > 0.0 else 1.0,
            max_new_tokens=kwargs["max_new_tokens"],
            top_p=kwargs["top_p"],
            repetition_penalty=kwargs["repetition_penalty"],
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    new_tokens = output_ids[0][input_len:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

_SEP = "─" * 60

def print_single(prompt: str, completion: str):
    print(_SEP)
    print(f"Prompt : {prompt}")
    print(f"Output : {completion}")
    print(_SEP)


def print_comparison(prompt: str, base_out: str, ft_out: str):
    print(_SEP)
    print(f"Prompt      : {prompt}")
    print(_SEP)
    print(f"[Zero-shot]  {base_out}")
    print(_SEP)
    print(f"[Fine-tuned] {ft_out}")
    print(_SEP)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Text completion inference / comparison")
    p.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                   help="HF model id or local path (base model)")
    p.add_argument("--lora_dir", default=None,
                   help="LoRA checkpoint dir. When set, runs side-by-side comparison.")
    p.add_argument("--prompt", default=None,
                   help="Prompt to complete. Omit for interactive mode.")
    p.add_argument("--max_new_tokens", type=int, default=200)
    p.add_argument("--temperature", type=float, default=0.7,
                   help="Sampling temperature (0 = greedy)")
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--repetition_penalty", type=float, default=1.1)
    p.add_argument("--dtype", default="bfloat16",
                   choices=["bfloat16", "float16", "float32"])
    return p.parse_args()


def main():
    args = parse_args()
    compare = args.lora_dir is not None

    gen_kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
    )

    print(f"Loading base model '{args.model}' ...", flush=True)
    base_model, tokenizer = load_base_model(args.model, args.dtype)

    ft_model = None
    if compare:
        print(f"Loading fine-tuned model from '{args.lora_dir}' ...", flush=True)
        ft_model, _ = load_finetuned_model(args.model, args.lora_dir, args.dtype)

    print("Ready.\n")

    def run(prompt: str):
        base_out = generate(base_model, tokenizer, prompt, **gen_kwargs)
        if compare:
            ft_out = generate(ft_model, tokenizer, prompt, **gen_kwargs)
            print_comparison(prompt, base_out, ft_out)
        else:
            print_single(prompt, base_out)

    if args.prompt:
        run(args.prompt)
    else:
        mode = "comparison" if compare else "single-model"
        print(f"Interactive {mode} mode — empty line or Ctrl-C to quit.")
        while True:
            try:
                prompt = input(">>> ").strip()
            except (KeyboardInterrupt, EOFError):
                print()
                break
            if not prompt:
                break
            run(prompt)
            print()


if __name__ == "__main__":
    main()
