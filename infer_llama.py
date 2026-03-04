#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
Text-completion inference for a (fine-tuned) Llama / TinyLlama model.

Usage examples
--------------
# Base model, greedy decoding:
python infer_llama.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --prompt "오늘 날씨가"

# Fine-tuned LoRA checkpoint:
python infer_llama.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --lora_dir runs/llama_korean/checkpoint-500 \
    --prompt "서울은"

# Interactive REPL (leave --prompt empty):
python infer_llama.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0
"""

import argparse
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_model(model_name: str, lora_dir: str | None, dtype: str):
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

    if lora_dir:
        from peft import PeftModel  # type: ignore
        model = PeftModel.from_pretrained(model, lora_dir)
        model = model.merge_and_unload()  # fold LoRA weights for faster inference

    model.eval()
    return model, tokenizer


def generate(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
) -> str:
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_len = inputs["input_ids"].shape[-1]

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0.0,
            temperature=temperature if temperature > 0.0 else 1.0,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Return only the newly generated tokens (strip the prompt)
    new_tokens = output_ids[0][input_len:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Text completion inference")
    p.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                   help="HF model id or local directory")
    p.add_argument("--lora_dir", default=None,
                   help="Path to a saved LoRA / PEFT checkpoint (optional)")
    p.add_argument("--prompt", default=None,
                   help="Input text to complete. Omit for interactive mode.")
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

    print(f"Loading model '{args.model}' ...", flush=True)
    model, tokenizer = load_model(args.model, args.lora_dir, args.dtype)
    print("Model ready.\n")

    gen_kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
    )

    if args.prompt:
        # Single-shot mode
        completion = generate(model, tokenizer, args.prompt, **gen_kwargs)
        print(f"Prompt:     {args.prompt}")
        print(f"Completion: {completion}")
    else:
        # Interactive REPL
        print("Interactive mode — enter your prompt (Ctrl-C or empty line to quit).")
        while True:
            try:
                prompt = input(">>> ").strip()
            except (KeyboardInterrupt, EOFError):
                print()
                break
            if not prompt:
                break
            completion = generate(model, tokenizer, prompt, **gen_kwargs)
            print(completion)
            print()


if __name__ == "__main__":
    main()
