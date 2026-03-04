#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
Fine-tune a Llama language model on Korean text using LoRA / QLoRA.

Supports:
  - Any Llama-family model available on Hugging Face Hub
  - Parameter-efficient fine-tuning via LoRA (PEFT)
  - Optional 4-bit quantization (QLoRA) via bitsandbytes
  - Causal language modelling objective (next-token prediction)
  - Korean text datasets: local text / JSONL files or HF Hub datasets
  - Gradient checkpointing, gradient accumulation, mixed-precision (bf16/fp16)
  - Checkpoint saving / resuming

Dataset formats
---------------
1. Plain-text file  (.txt): one document per line.
2. JSONL file (.jsonl): each line is {"text": "..."}.
3. Hugging Face dataset: specify ``data.hf_dataset_name`` (e.g. ``"wikimedia/wikipedia"``
   with ``data.hf_dataset_config="20231101.ko"``).

Single-GPU launch
-----------------
    python finetune_llama_korean.py \\
        model.name=meta-llama/Llama-3.2-1B \\
        data.train_file=data/korean_train.txt \\
        train.output_dir=runs/llama_korean

Multi-GPU launch (accelerate)
------------------------------
    accelerate launch --num_processes 4 finetune_llama_korean.py \\
        model.name=meta-llama/Llama-3.2-1B \\
        data.hf_dataset_name=wikimedia/wikipedia \\
        data.hf_dataset_config=20231101.ko \\
        train.output_dir=runs/llama_korean_wiki

YAML config launch
------------------
    python finetune_llama_korean.py config=configs/llama_korean.yaml
"""

import json
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ModelArgs:
    """Model and LoRA configuration."""
    # Hugging Face model id or local directory
    name: str = "meta-llama/Llama-3.2-1B"
    # Use 4-bit quantization (QLoRA); requires bitsandbytes
    load_in_4bit: bool = False
    # Use 8-bit quantization; requires bitsandbytes
    load_in_8bit: bool = False
    # torch dtype for non-quantised weights ("auto", "float16", "bfloat16")
    torch_dtype: str = "bfloat16"
    # HF auth token for gated models (or set HF_TOKEN env var)
    hf_token: Optional[str] = None

    # LoRA hyper-parameters
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    # Modules to apply LoRA to; "auto" selects q/v projection layers
    lora_target_modules: str = "auto"


@dataclass
class DataArgs:
    """Dataset and tokenisation configuration."""
    # --- local file source (choose one of the three) ---
    # Plain .txt (one doc per line) or .jsonl with {"text": "..."}
    train_file: str = ""
    val_file: str = ""

    # --- Hugging Face Hub dataset source ---
    hf_dataset_name: str = ""          # e.g. "wikimedia/wikipedia"
    hf_dataset_config: str = ""        # e.g. "20231101.ko"
    hf_text_column: str = "text"       # column containing the Korean text
    hf_val_split_pct: float = 0.02     # fraction to use as val when no val split exists

    # Tokenisation
    max_seq_length: int = 512          # tokens per training example
    # Strategy for long documents: "truncate" or "chunk"
    overflow_strategy: str = "chunk"

    # DataLoader
    batch_size: int = 4
    num_workers: int = 2
    pin_memory: bool = True


@dataclass
class TrainArgs:
    """Training loop configuration."""
    output_dir: str = "runs/llama_korean"
    seed: int = 42

    num_epochs: int = 3
    max_steps: int = -1               # overrides num_epochs when > 0
    grad_accum_steps: int = 8
    max_grad_norm: float = 1.0

    learning_rate: float = 2e-4
    lr_scheduler: str = "cosine"      # "cosine" | "linear" | "constant"
    warmup_ratio: float = 0.03

    bf16: bool = True                 # mixed-precision bf16 (preferred on Ampere+)
    fp16: bool = False                # mixed-precision fp16
    gradient_checkpointing: bool = True

    log_freq: int = 10                # log every N steps
    save_freq: int = 500              # save checkpoint every N steps
    eval_freq: int = 200              # run validation every N steps (0 = skip)

    # Resume from a previously saved checkpoint directory
    resume_from: str = ""


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def _read_texts_from_file(path: str) -> List[str]:
    """Return a list of raw text strings from a .txt or .jsonl file."""
    texts: List[str] = []
    suffix = Path(path).suffix.lower()
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if suffix == ".jsonl":
                obj = json.loads(line)
                # Support {"text": ...} or {"content": ...}
                texts.append(obj.get("text") or obj.get("content", ""))
            else:
                texts.append(line)
    return texts


def _load_hf_texts(args: DataArgs) -> Dict[str, List[str]]:
    """Load text from a Hugging Face dataset and return train/val splits."""
    from datasets import load_dataset  # type: ignore

    logger.info(
        f"Loading HF dataset '{args.hf_dataset_name}' "
        f"config='{args.hf_dataset_config}'"
    )
    ds = load_dataset(
        args.hf_dataset_name,
        args.hf_dataset_config or None,
        trust_remote_code=True,
    )

    # Many datasets have a "train" split; create a val split when missing
    if "validation" in ds:
        train_texts = [ex[args.hf_text_column] for ex in ds["train"]]
        val_texts = [ex[args.hf_text_column] for ex in ds["validation"]]
    else:
        split = ds["train"].train_test_split(
            test_size=args.hf_val_split_pct, seed=42
        )
        train_texts = [ex[args.hf_text_column] for ex in split["train"]]
        val_texts = [ex[args.hf_text_column] for ex in split["test"]]

    return {"train": train_texts, "val": val_texts}


class KoreanTextDataset(Dataset):
    """
    Token-level dataset for causal language modelling.

    Each raw text string is tokenised, then either truncated or chunked into
    non-overlapping windows of ``max_seq_length`` tokens.  The ``labels``
    tensor is a shifted copy of ``input_ids`` so the model learns to predict
    the next token; padding positions use label id -100 (ignored by
    cross-entropy).

    Args:
        texts:          List of raw Korean strings.
        tokenizer:      A Hugging Face ``PreTrainedTokenizer``.
        max_seq_length: Maximum number of tokens per training example.
        overflow:       "truncate" — keep only the first ``max_seq_length``
                        tokens of each text.
                        "chunk"   — split each text into as many
                        ``max_seq_length``-token windows as needed.
    """

    def __init__(
        self,
        texts: List[str],
        tokenizer,
        max_seq_length: int = 512,
        overflow: str = "chunk",
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.examples: List[Dict[str, torch.Tensor]] = []

        for text in texts:
            ids = tokenizer.encode(text, add_special_tokens=True)
            if overflow == "chunk":
                # Split into non-overlapping windows
                for start in range(0, max(1, len(ids) - 1), max_seq_length):
                    chunk = ids[start : start + max_seq_length]
                    if len(chunk) < 2:        # need at least 2 tokens for a label
                        continue
                    self.examples.append(self._make_example(chunk))
            else:
                # Just truncate
                chunk = ids[:max_seq_length]
                if len(chunk) >= 2:
                    self.examples.append(self._make_example(chunk))

        logger.info(f"Built {len(self.examples):,} training examples")

    def _make_example(self, ids: List[int]) -> Dict[str, torch.Tensor]:
        """Pad / truncate a token sequence and build (input_ids, labels, mask)."""
        pad_id = self.tokenizer.pad_token_id or 0
        length = len(ids)
        pad_len = self.max_seq_length - length

        input_ids = ids + [pad_id] * pad_len
        labels = ids + [-100] * pad_len       # ignore padding in loss
        attention_mask = [1] * length + [0] * pad_len

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.examples[idx]


def _collate(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    return {k: torch.stack([ex[k] for ex in batch]) for k in batch[0]}


def build_dataloaders(
    data_args: DataArgs, tokenizer
) -> Dict[str, Optional[DataLoader]]:
    """Build train and (optionally) val DataLoaders."""
    if data_args.hf_dataset_name:
        splits = _load_hf_texts(data_args)
        train_texts = splits["train"]
        val_texts = splits["val"]
    else:
        if not data_args.train_file:
            raise ValueError(
                "Provide data.train_file (local path) or data.hf_dataset_name."
            )
        train_texts = _read_texts_from_file(data_args.train_file)
        val_texts = (
            _read_texts_from_file(data_args.val_file)
            if data_args.val_file
            else []
        )

    def _make_loader(texts: List[str], shuffle: bool) -> Optional[DataLoader]:
        if not texts:
            return None
        ds = KoreanTextDataset(
            texts,
            tokenizer,
            max_seq_length=data_args.max_seq_length,
            overflow=data_args.overflow_strategy,
        )
        return DataLoader(
            ds,
            batch_size=data_args.batch_size,
            shuffle=shuffle,
            num_workers=data_args.num_workers,
            pin_memory=data_args.pin_memory,
            collate_fn=_collate,
            drop_last=shuffle,
        )

    return {
        "train": _make_loader(train_texts, shuffle=True),
        "val": _make_loader(val_texts, shuffle=False),
    }


# ---------------------------------------------------------------------------
# Model + LoRA setup
# ---------------------------------------------------------------------------

def _resolve_dtype(name: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
        "auto": torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    }
    return mapping.get(name, torch.bfloat16)


def build_model_and_tokenizer(args: ModelArgs):
    """
    Load a Llama model and tokenizer, then optionally wrap with LoRA adapters.

    Quantisation note
    -----------------
    When ``load_in_4bit=True`` the model is loaded in NF4 format (QLoRA).
    The LoRA adapters are kept in full precision so only a small fraction of
    parameters are trained, keeping GPU memory usage low.

    Returns:
        model:     (Peft)Model ready for training
        tokenizer: Corresponding tokenizer
    """
    from transformers import (  # type: ignore
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )

    token = args.hf_token or os.environ.get("HF_TOKEN")
    dtype = _resolve_dtype(args.torch_dtype)

    # --- Quantisation config ---
    bnb_config = None
    if args.load_in_4bit:
        from transformers import BitsAndBytesConfig  # noqa: F811
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
        )
    elif args.load_in_8bit:
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)

    logger.info(f"Loading model '{args.name}' ...")
    model = AutoModelForCausalLM.from_pretrained(
        args.name,
        quantization_config=bnb_config,
        torch_dtype=dtype if bnb_config is None else None,
        device_map="auto",
        token=token,
        trust_remote_code=True,
    )

    # --- Tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(
        args.name,
        token=token,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"   # required for causal LM training

    # --- LoRA ---
    if args.use_lora:
        from peft import LoraConfig, TaskType, get_peft_model  # type: ignore
        from peft import prepare_model_for_kbit_training          # type: ignore

        if args.load_in_4bit or args.load_in_8bit:
            model = prepare_model_for_kbit_training(model)

        target_modules = (
            _auto_lora_targets(model)
            if args.lora_target_modules == "auto"
            else args.lora_target_modules.split(",")
        )
        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=target_modules,
            bias="none",
        )
        model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()
    else:
        _log_param_count(model)

    return model, tokenizer


def _auto_lora_targets(model) -> List[str]:
    """Heuristically pick query/value projection names for LoRA."""
    candidates = {"q_proj", "v_proj", "k_proj", "o_proj", "gate_proj",
                  "up_proj", "down_proj"}
    found = {
        name.split(".")[-1]
        for name, _ in model.named_modules()
        if name.split(".")[-1] in candidates
    }
    # Prefer q/v only for a smaller adapter; fall back to all linear layers
    preferred = {"q_proj", "v_proj"} & found
    return sorted(preferred or found) or ["q_proj", "v_proj"]


def _log_param_count(model) -> None:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        f"Parameters: total={total:,}  trainable={trainable:,} "
        f"({100 * trainable / max(total, 1):.2f}%)"
    )


# ---------------------------------------------------------------------------
# LR scheduler
# ---------------------------------------------------------------------------

def _build_scheduler(optimizer, name: str, num_training_steps: int,
                     warmup_steps: int):
    from transformers import get_scheduler  # type: ignore
    return get_scheduler(
        name=name,
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_training_steps,
    )


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(
    model,
    optimizer,
    scheduler,
    step: int,
    output_dir: Path,
    tokenizer=None,
) -> None:
    ckpt_dir = output_dir / f"checkpoint-{step}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Save adapter weights (PEFT) or full weights
    if hasattr(model, "save_pretrained"):
        model.save_pretrained(ckpt_dir)
    else:
        torch.save(model.state_dict(), ckpt_dir / "model.pt")

    if tokenizer is not None:
        tokenizer.save_pretrained(ckpt_dir)

    torch.save(
        {"optimizer": optimizer.state_dict(),
         "scheduler": scheduler.state_dict(),
         "step": step},
        ckpt_dir / "trainer_state.pt",
    )
    logger.info(f"Checkpoint saved → {ckpt_dir}")


def _latest_checkpoint(output_dir: Path) -> Optional[Path]:
    """Return the most recently saved checkpoint directory, or None."""
    ckpts = sorted(output_dir.glob("checkpoint-*"), key=os.path.getmtime)
    return ckpts[-1] if ckpts else None


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    """Compute average cross-entropy loss on the validation set."""
    model.eval()
    total_loss, n_batches = 0.0, 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(**batch)
        total_loss += out.loss.item()
        n_batches += 1
    model.train()
    return {"val/loss": total_loss / max(n_batches, 1)}


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(model_args: ModelArgs, data_args: DataArgs, train_args: TrainArgs) -> None:
    # ------------------------------------------------------------------
    # Device
    # ------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # ------------------------------------------------------------------
    # Reproducibility
    # ------------------------------------------------------------------
    import random, numpy as np  # noqa: E401
    random.seed(train_args.seed)
    np.random.seed(train_args.seed)
    torch.manual_seed(train_args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(train_args.seed)

    # ------------------------------------------------------------------
    # Output directory
    # ------------------------------------------------------------------
    output_dir = Path(train_args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Model & tokenizer
    # ------------------------------------------------------------------
    model, tokenizer = build_model_and_tokenizer(model_args)

    if train_args.gradient_checkpointing:
        model.enable_input_require_grads()   # required for PEFT + grad ckpt
        model.gradient_checkpointing_enable()

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    loaders = build_dataloaders(data_args, tokenizer)
    train_loader = loaders["train"]
    val_loader = loaders["val"]

    if train_loader is None:
        raise ValueError("Training dataloader is empty. Check your data config.")

    # ------------------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------------------
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=train_args.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0.01,
    )

    steps_per_epoch = len(train_loader)
    if train_args.max_steps > 0:
        total_steps = train_args.max_steps
        num_epochs = (total_steps + steps_per_epoch - 1) // steps_per_epoch
    else:
        num_epochs = train_args.num_epochs
        total_steps = num_epochs * steps_per_epoch

    # Account for gradient accumulation in scheduler steps
    effective_total = total_steps // max(train_args.grad_accum_steps, 1)
    warmup_steps = int(effective_total * train_args.warmup_ratio)

    scheduler = _build_scheduler(
        optimizer,
        name=train_args.lr_scheduler,
        num_training_steps=effective_total,
        warmup_steps=warmup_steps,
    )

    # ------------------------------------------------------------------
    # Mixed-precision scaler
    # ------------------------------------------------------------------
    use_amp = (train_args.fp16 or train_args.bf16) and device.type == "cuda"
    amp_dtype = (
        torch.bfloat16 if train_args.bf16 else torch.float16
    )
    scaler = (
        torch.cuda.amp.GradScaler()
        if use_amp and amp_dtype == torch.float16
        else None
    )

    # ------------------------------------------------------------------
    # Resume
    # ------------------------------------------------------------------
    start_step = 0
    if train_args.resume_from:
        ckpt = Path(train_args.resume_from)
    else:
        ckpt = _latest_checkpoint(output_dir)

    if ckpt is not None and (ckpt / "trainer_state.pt").exists():
        state = torch.load(ckpt / "trainer_state.pt", map_location="cpu")
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_step = state["step"] + 1
        logger.info(f"Resumed from step {start_step} (checkpoint: {ckpt})")

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    metrics_log = output_dir / "metrics.jsonl"
    model.train()
    global_step = 0
    optimizer.zero_grad()

    logger.info(
        f"Starting training: epochs={num_epochs}  steps/epoch={steps_per_epoch}  "
        f"total_steps={total_steps}  effective_optimizer_steps={effective_total}"
    )

    for epoch in range(num_epochs):
        for batch in train_loader:
            if train_args.max_steps > 0 and global_step >= train_args.max_steps:
                break

            if global_step < start_step:
                global_step += 1
                continue

            batch = {k: v.to(device) for k, v in batch.items()}

            # Forward
            if use_amp:
                with torch.cuda.amp.autocast(dtype=amp_dtype):
                    out = model(**batch)
                    loss = out.loss / train_args.grad_accum_steps
            else:
                out = model(**batch)
                loss = out.loss / train_args.grad_accum_steps

            # Backward
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            # Optimizer step (every grad_accum_steps)
            if (global_step + 1) % train_args.grad_accum_steps == 0:
                if train_args.max_grad_norm > 0:
                    if scaler is not None:
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), train_args.max_grad_norm
                    )
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            # Logging
            if global_step % train_args.log_freq == 0:
                lr = scheduler.get_last_lr()[0]
                raw_loss = loss.item() * train_args.grad_accum_steps
                msg = (
                    f"epoch={epoch+1}  step={global_step:6d}  "
                    f"loss={raw_loss:.4f}  lr={lr:.2e}"
                )
                logger.info(msg)
                with open(metrics_log, "a") as f:
                    f.write(json.dumps(
                        {"step": global_step, "epoch": epoch + 1,
                         "train/loss": raw_loss, "lr": lr}
                    ) + "\n")

            # Validation
            if (
                val_loader is not None
                and train_args.eval_freq > 0
                and global_step > 0
                and global_step % train_args.eval_freq == 0
            ):
                val_metrics = evaluate(model, val_loader, device)
                val_metrics["step"] = global_step
                logger.info(f"val  loss={val_metrics['val/loss']:.4f}")
                with open(metrics_log, "a") as f:
                    f.write(json.dumps(val_metrics) + "\n")

            # Checkpoint
            if (
                global_step > 0
                and train_args.save_freq > 0
                and global_step % train_args.save_freq == 0
            ):
                save_checkpoint(
                    model, optimizer, scheduler, global_step,
                    output_dir, tokenizer
                )

            global_step += 1

        if train_args.max_steps > 0 and global_step >= train_args.max_steps:
            break

    # Final save
    save_checkpoint(
        model, optimizer, scheduler, global_step - 1,
        output_dir, tokenizer
    )
    logger.info(f"Training complete. Model saved to {output_dir}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Config priority (highest → lowest):
        1. CLI key=value overrides  (e.g. model.name=... train.bf16=true)
        2. --config path/to/config.yaml
        3. Dataclass defaults
    """
    cli = OmegaConf.from_cli()

    if "config" in cli:
        config_file = cli.pop("config")
        base_cfg = OmegaConf.load(config_file)
    else:
        base_cfg = OmegaConf.merge(
            OmegaConf.structured(ModelArgs()),
            OmegaConf.structured(DataArgs()),
            OmegaConf.structured(TrainArgs()),
        )
        base_cfg = OmegaConf.create({
            "model": OmegaConf.structured(ModelArgs()),
            "data": OmegaConf.structured(DataArgs()),
            "train": OmegaConf.structured(TrainArgs()),
        })

    cfg = OmegaConf.merge(base_cfg, cli)

    model_args: ModelArgs = OmegaConf.to_object(
        OmegaConf.merge(OmegaConf.structured(ModelArgs()), cfg.get("model", {}))
    )
    data_args: DataArgs = OmegaConf.to_object(
        OmegaConf.merge(OmegaConf.structured(DataArgs()), cfg.get("data", {}))
    )
    train_args: TrainArgs = OmegaConf.to_object(
        OmegaConf.merge(OmegaConf.structured(TrainArgs()), cfg.get("train", {}))
    )

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )

    logger.info("=== Llama Korean Fine-Tuning ===")
    logger.info(f"Model : {model_args.name}")
    logger.info(f"LoRA  : {model_args.use_lora}  (r={model_args.lora_r}, alpha={model_args.lora_alpha})")
    logger.info(f"4-bit : {model_args.load_in_4bit}   8-bit: {model_args.load_in_8bit}")
    logger.info(f"Output: {train_args.output_dir}")

    train(model_args, data_args, train_args)


if __name__ == "__main__":
    main()
