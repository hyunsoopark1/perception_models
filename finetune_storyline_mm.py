#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
Fine-tune a vision-language model (VLM) on the storyline_mm dataset.

The model learns to generate a text story conditioned on 1–3 images.

Data layout expected (produced by download_storyline_mm.py)
-----------------------------------------------------------
  ~/data/storyline_mm/
    0000001/
      text.txt          ← target story text
      image_0.<ext>     ← required
      image_1.<ext>     ← optional
      image_2.<ext>     ← optional
    0000002/
      ...

Supported models
----------------
Any Hugging Face VLM that uses AutoProcessor + AutoModelForVision2Seq /
MllamaForConditionalGeneration, e.g.:
  - meta-llama/Llama-3.2-11B-Vision-Instruct   (default)
  - meta-llama/Llama-3.2-11B-Vision
  - llava-hf/llava-1.5-7b-hf
  - Qwen/Qwen2-VL-7B-Instruct

Single-GPU launch
-----------------
  python finetune_storyline_mm.py \\
      model.name=meta-llama/Llama-3.2-11B-Vision-Instruct \\
      data.data_dir=~/data/storyline_mm \\
      train.output_dir=runs/storyline_mm

Multi-GPU (accelerate)
----------------------
  accelerate launch --num_processes 4 finetune_storyline_mm.py \\
      model.name=meta-llama/Llama-3.2-11B-Vision-Instruct \\
      data.data_dir=~/data/storyline_mm \\
      train.output_dir=runs/storyline_mm

YAML config
-----------
  python finetune_storyline_mm.py config=configs/storyline_mm.yaml
"""

import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from omegaconf import OmegaConf
from PIL import Image
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class ModelArgs:
    name: str = "meta-llama/Llama-3.2-11B-Vision-Instruct"
    load_in_4bit: bool = False
    load_in_8bit: bool = False
    torch_dtype: str = "bfloat16"
    hf_token: Optional[str] = None

    # LoRA
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    # "auto" = heuristic; or comma-separated module names
    lora_target_modules: str = "auto"


@dataclass
class DataArgs:
    data_dir: str = "~/data/storyline_mm"
    val_split: float = 0.05          # fraction of folders used for validation
    max_seq_length: int = 1024       # max tokens for text (images handled separately)
    # Prompt sent to the model before the target text
    instruction: str = "Look at the image(s) and write a descriptive story."

    batch_size: int = 2
    num_workers: int = 2


@dataclass
class TrainArgs:
    output_dir: str = "runs/storyline_mm"
    seed: int = 42

    num_epochs: int = 3
    max_steps: int = -1
    grad_accum_steps: int = 8
    max_grad_norm: float = 1.0

    learning_rate: float = 2e-4
    lr_scheduler: str = "cosine"
    warmup_ratio: float = 0.03

    bf16: bool = True
    fp16: bool = False
    gradient_checkpointing: bool = True

    log_freq: int = 10
    save_freq: int = 200
    eval_freq: int = 100

    resume_from: str = ""


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def _find_images(folder: Path) -> List[Path]:
    """Return image_0, image_1, image_2 paths that exist in the folder."""
    images = []
    for idx in range(3):
        for ext in IMAGE_EXTENSIONS:
            p = folder / f"image_{idx}{ext}"
            if p.exists():
                images.append(p)
                break
    return images


def _load_folders(data_dir: Path) -> List[Path]:
    """Return sorted list of sample folders (must contain text.txt + ≥1 image)."""
    folders = []
    for folder in sorted(data_dir.iterdir()):
        if not folder.is_dir():
            continue
        if not (folder / "text.txt").exists():
            continue
        if not _find_images(folder):
            continue
        folders.append(folder)
    return folders


class StorylineMMDataset(Dataset):
    """
    Each sample: 1–3 images + a target story text.

    The model is given a user turn containing the image(s) and an instruction,
    then trained to produce the target text as the assistant turn.

    Labels are masked (-100) over the instruction/image tokens so that loss is
    computed only on the target text tokens.
    """

    def __init__(
        self,
        folders: List[Path],
        processor,
        instruction: str,
        max_seq_length: int,
    ):
        self.folders = folders
        self.processor = processor
        self.instruction = instruction
        self.max_seq_length = max_seq_length

    def __len__(self) -> int:
        return len(self.folders)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        folder = self.folders[idx]
        text = (folder / "text.txt").read_text(encoding="utf-8").strip()
        image_paths = _find_images(folder)
        images = [Image.open(p).convert("RGB") for p in image_paths]

        # Build conversation: user (images + instruction) → assistant (story)
        user_content = [{"type": "image"} for _ in images]
        user_content.append({"type": "text", "text": self.instruction})

        messages = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": [{"type": "text", "text": text}]},
        ]

        # apply_chat_template builds the full prompt string
        full_prompt = self.processor.apply_chat_template(
            messages, add_generation_prompt=False, tokenize=False
        )
        # Prompt up to (not including) the assistant response — for label masking
        prompt_only = self.processor.apply_chat_template(
            [messages[0]], add_generation_prompt=True, tokenize=False
        )

        # Tokenize full sequence
        enc = self.processor(
            text=full_prompt,
            images=images,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_seq_length,
        )

        input_ids = enc["input_ids"][0]
        attention_mask = enc["attention_mask"][0]

        # Mask labels: set prompt tokens to -100
        prompt_len = len(
            self.processor.tokenizer(
                prompt_only, add_special_tokens=False
            )["input_ids"]
        )
        labels = input_ids.clone()
        labels[:prompt_len] = -100  # ignore prompt + image tokens in loss

        sample = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

        # pixel_values / image_sizes are present when images were supplied
        if "pixel_values" in enc:
            sample["pixel_values"] = enc["pixel_values"][0]
        if "image_sizes" in enc:
            sample["image_sizes"] = enc["image_sizes"][0]

        return sample


def _collate(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Pad sequences to the longest in the batch."""
    keys = batch[0].keys()
    out = {}
    for k in keys:
        tensors = [ex[k] for ex in batch]
        if k in {"input_ids", "attention_mask", "labels"}:
            # Left-pad to max length (right-pad also works; right is simpler)
            max_len = max(t.shape[0] for t in tensors)
            pad_id = 0 if k != "labels" else -100
            padded = torch.full((len(tensors), max_len), pad_id, dtype=tensors[0].dtype)
            for i, t in enumerate(tensors):
                padded[i, : t.shape[0]] = t
            out[k] = padded
        else:
            # pixel_values / image_sizes: stack if same shape, else leave as list
            try:
                out[k] = torch.stack(tensors)
            except RuntimeError:
                out[k] = tensors  # variable shape — model handles list
    return out


def build_dataloaders(data_args: DataArgs, processor) -> Dict[str, Optional[DataLoader]]:
    data_dir = Path(data_args.data_dir).expanduser()
    folders = _load_folders(data_dir)
    logger.info(f"Found {len(folders)} samples in {data_dir}")

    n_val = max(1, int(len(folders) * data_args.val_split))
    val_folders = folders[-n_val:]
    train_folders = folders[:-n_val]
    logger.info(f"Train: {len(train_folders)}  Val: {len(val_folders)}")

    def _make(fols, shuffle):
        if not fols:
            return None
        ds = StorylineMMDataset(
            fols, processor, data_args.instruction, data_args.max_seq_length
        )
        return DataLoader(
            ds,
            batch_size=data_args.batch_size,
            shuffle=shuffle,
            num_workers=data_args.num_workers,
            collate_fn=_collate,
            pin_memory=True,
        )

    return {"train": _make(train_folders, True), "val": _make(val_folders, False)}


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def _resolve_dtype(name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
        "auto": torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    }.get(name, torch.bfloat16)


def build_model_and_processor(args: ModelArgs):
    from transformers import AutoProcessor  # type: ignore
    try:
        from transformers import AutoModelForVision2Seq  # type: ignore
    except ImportError:
        from transformers import AutoModel as AutoModelForVision2Seq  # type: ignore

    token = args.hf_token or os.environ.get("HF_TOKEN")
    dtype = _resolve_dtype(args.torch_dtype)

    bnb_config = None
    if args.load_in_4bit or args.load_in_8bit:
        from transformers import BitsAndBytesConfig  # type: ignore
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=args.load_in_4bit,
            load_in_8bit=args.load_in_8bit,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
        )

    processor = AutoProcessor.from_pretrained(
        args.name, token=token, trust_remote_code=True
    )
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    processor.tokenizer.padding_side = "right"

    model = AutoModelForVision2Seq.from_pretrained(
        args.name,
        quantization_config=bnb_config,
        torch_dtype=dtype if bnb_config is None else None,
        device_map="auto",
        token=token,
        trust_remote_code=True,
    )

    if args.use_lora:
        from peft import LoraConfig, TaskType, get_peft_model  # type: ignore
        from peft import prepare_model_for_kbit_training       # type: ignore

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

    return model, processor


def _auto_lora_targets(model) -> List[str]:
    candidates = {
        "q_proj", "v_proj", "k_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
        # Vision cross-attention layers (Llama 3.2 Vision)
        "cross_attn_q", "cross_attn_k", "cross_attn_v", "cross_attn_o",
    }
    found = {
        name.split(".")[-1]
        for name, _ in model.named_modules()
        if name.split(".")[-1] in candidates
    }
    preferred = {"q_proj", "v_proj"} & found
    return sorted(preferred or found) or ["q_proj", "v_proj"]


# ---------------------------------------------------------------------------
# Checkpoint helpers  (same as finetune_llama_korean.py)
# ---------------------------------------------------------------------------

def save_checkpoint(model, optimizer, scheduler, step, output_dir, processor):
    ckpt_dir = output_dir / f"checkpoint-{step}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    if hasattr(model, "save_pretrained"):
        model.save_pretrained(ckpt_dir)
    else:
        torch.save(model.state_dict(), ckpt_dir / "model.pt")
    processor.save_pretrained(ckpt_dir)
    torch.save(
        {"optimizer": optimizer.state_dict(),
         "scheduler": scheduler.state_dict(),
         "step": step},
        ckpt_dir / "trainer_state.pt",
    )
    logger.info(f"Checkpoint saved → {ckpt_dir}")


def _latest_checkpoint(output_dir: Path) -> Optional[Path]:
    ckpts = sorted(output_dir.glob("checkpoint-*"), key=os.path.getmtime)
    return ckpts[-1] if ckpts else None


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, loader, device) -> Dict[str, float]:
    model.eval()
    total_loss, n = 0.0, 0
    for batch in loader:
        batch = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        out = model(**batch)
        total_loss += out.loss.item()
        n += 1
    model.train()
    return {"val/loss": total_loss / max(n, 1)}


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(model_args: ModelArgs, data_args: DataArgs, train_args: TrainArgs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    import random, numpy as np  # noqa: E401
    random.seed(train_args.seed)
    np.random.seed(train_args.seed)
    torch.manual_seed(train_args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(train_args.seed)

    output_dir = Path(train_args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading model '{model_args.name}' ...")
    model, processor = build_model_and_processor(model_args)

    if train_args.gradient_checkpointing:
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable()

    loaders = build_dataloaders(data_args, processor)
    train_loader = loaders["train"]
    val_loader = loaders["val"]
    if train_loader is None:
        raise ValueError("No training samples found. Check data.data_dir.")

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

    effective_total = total_steps // max(train_args.grad_accum_steps, 1)
    warmup_steps = int(effective_total * train_args.warmup_ratio)

    from transformers import get_scheduler  # type: ignore
    scheduler = get_scheduler(
        train_args.lr_scheduler, optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=effective_total,
    )

    use_amp = (train_args.fp16 or train_args.bf16) and device.type == "cuda"
    amp_dtype = torch.bfloat16 if train_args.bf16 else torch.float16
    scaler = (
        torch.cuda.amp.GradScaler()
        if use_amp and amp_dtype == torch.float16 else None
    )

    # Resume
    start_step = 0
    ckpt = Path(train_args.resume_from) if train_args.resume_from else _latest_checkpoint(output_dir)
    if ckpt and (ckpt / "trainer_state.pt").exists():
        state = torch.load(ckpt / "trainer_state.pt", map_location="cpu")
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_step = state["step"] + 1
        logger.info(f"Resumed from step {start_step} ({ckpt})")

    metrics_log = output_dir / "metrics.jsonl"
    model.train()
    global_step = 0
    optimizer.zero_grad()

    logger.info(
        f"Training: epochs={num_epochs}  steps/epoch={steps_per_epoch}  "
        f"total={total_steps}  effective={effective_total}"
    )

    for epoch in range(num_epochs):
        for batch in train_loader:
            if train_args.max_steps > 0 and global_step >= train_args.max_steps:
                break
            if global_step < start_step:
                global_step += 1
                continue

            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            if use_amp:
                with torch.cuda.amp.autocast(dtype=amp_dtype):
                    loss = model(**batch).loss / train_args.grad_accum_steps
            else:
                loss = model(**batch).loss / train_args.grad_accum_steps

            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (global_step + 1) % train_args.grad_accum_steps == 0:
                if train_args.max_grad_norm > 0:
                    if scaler:
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), train_args.max_grad_norm)
                if scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if global_step % train_args.log_freq == 0:
                lr = scheduler.get_last_lr()[0]
                raw_loss = loss.item() * train_args.grad_accum_steps
                logger.info(
                    f"epoch={epoch+1}  step={global_step:6d}  loss={raw_loss:.4f}  lr={lr:.2e}"
                )
                with open(metrics_log, "a") as f:
                    f.write(json.dumps(
                        {"step": global_step, "epoch": epoch + 1,
                         "train/loss": raw_loss, "lr": lr}
                    ) + "\n")

            if (val_loader and train_args.eval_freq > 0
                    and global_step > 0 and global_step % train_args.eval_freq == 0):
                val_metrics = evaluate(model, val_loader, device)
                val_metrics["step"] = global_step
                logger.info(f"val  loss={val_metrics['val/loss']:.4f}")
                with open(metrics_log, "a") as f:
                    f.write(json.dumps(val_metrics) + "\n")

            if (global_step > 0 and train_args.save_freq > 0
                    and global_step % train_args.save_freq == 0):
                save_checkpoint(model, optimizer, scheduler, global_step, output_dir, processor)

            global_step += 1

        if train_args.max_steps > 0 and global_step >= train_args.max_steps:
            break

    save_checkpoint(model, optimizer, scheduler, global_step - 1, output_dir, processor)
    logger.info(f"Done. Model saved to {output_dir}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    cli = OmegaConf.from_cli()

    if "config" in cli:
        base_cfg = OmegaConf.load(cli.pop("config"))
    else:
        base_cfg = OmegaConf.create({
            "model": OmegaConf.structured(ModelArgs()),
            "data":  OmegaConf.structured(DataArgs()),
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

    logger.info("=== Storyline Multi-Modal Fine-Tuning ===")
    logger.info(f"Model : {model_args.name}")
    logger.info(f"Data  : {data_args.data_dir}")
    logger.info(f"LoRA  : {model_args.use_lora}  (r={model_args.lora_r}, alpha={model_args.lora_alpha})")
    logger.info(f"Output: {train_args.output_dir}")

    train(model_args, data_args, train_args)


if __name__ == "__main__":
    main()
