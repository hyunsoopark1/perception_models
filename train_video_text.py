#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
Video-text contrastive fine-tuning for PE with spatiotemporal encoding.

Trains the TemporalTransformer (added to CLIP.encode_video) while keeping
the pretrained spatial image encoder frozen.  The training objective is the
symmetric InfoNCE contrastive loss between L2-normalised video embeddings
and paired text embeddings (identical to the original CLIP objective).

Dataset format — one JSON object per line (JSONL):
    {
        "video": "clip.mp4",                    # filename; joined with data.video_root
        "video_id": "optional_id",
        "source": "VATEX",
        "type": "captioning",
        "instruction": "...",
        "conversations": [
            {"from": "human",     "value": "..."},
            {"from": "assistant", "value": "caption text used for training"}
        ]
    }
    The caption is taken from the last "assistant" turn in conversations.

Single-GPU launch:
    python train_video_text.py data.train_data=data/train.jsonl

Multi-GPU launch (torchrun):
    torchrun --nproc_per_node=8 train_video_text.py \\
        data.train_data=data/train.jsonl \\
        data.val_data=data/val.jsonl \\
        model.pe_config=PE-Core-G14-448

YAML config launch:
    torchrun --nproc_per_node=8 train_video_text.py config=configs/video_text.yaml
"""

import json
import logging
import os
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from core.distributed import (
    DistributedArgs,
    EnvironmentArgs,
    get_global_rank,
    get_is_master,
    get_local_rank,
    get_world_size,
    setup_env,
    setup_torch_distributed,
)
from core.metrics import LoggingArgs, MetricLogger, log_model_params
from core.optim import OptimArgs, build_optimizer
from core.transforms.video_transform import VideoTransform
from core.vision_encoder.pe import CLIP
from core.vision_encoder.transforms import get_text_tokenizer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class DataArgs:
    """Dataset and data-loading configuration."""
    # Paths to JSONL files (conversations format, one JSON object per line)
    train_data: str = ""
    val_data: str = ""
    # Optional root directory prepended to each "video" filename in the JSONL.
    # Leave empty if the JSONL already contains absolute paths.
    video_root: str = ""

    image_size: int = 224       # Spatial resolution passed to the spatial encoder
    num_frames: int = 8         # Frames uniformly sampled per video clip
    sampling_fps: int = 1       # Target FPS when sampling frames
    batch_size: int = 32        # Per-GPU batch size
    num_workers: int = 4        # DataLoader worker processes
    pin_memory: bool = True


@dataclass
class ModelArgs:
    """Model configuration."""
    pe_config: str = "PE-Core-B16-224"  # Key into PE_VISION_CONFIG / PE_TEXT_CONFIG
    pretrained: bool = True              # Load pretrained spatial encoder from HF hub
    checkpoint_path: Optional[str] = None  # Local checkpoint; overrides hub download

    # Temporal encoder hyper-parameters
    num_temporal_layers: int = 2         # Depth of the TemporalTransformer
    num_temporal_heads: int = 8          # Attention heads in the TemporalTransformer

    # Training strategy
    freeze_spatial: bool = True          # Freeze visual + text towers; train only temporal
    freeze_logit_scale: bool = False     # Also freeze temperature; useful for eval ablation


@dataclass
class TrainArgs:
    """Top-level training configuration."""
    name: str = "pe_video_text"
    dump_dir: str = "runs/pe_video_text"
    seed: int = 42

    max_steps: int = 10_000
    grad_accum_steps: int = 1           # Gradient accumulation steps
    max_grad_norm: float = 1.0          # Gradient clipping; 0 = disabled

    log_freq: int = 20                  # Log training metrics every N steps
    save_freq: int = 1_000              # Save checkpoint every N steps
    eval_freq: int = 500                # Run validation every N steps (0 = no val)

    data: DataArgs = field(default_factory=DataArgs)
    model: ModelArgs = field(default_factory=ModelArgs)
    optim: OptimArgs = field(default_factory=OptimArgs)
    logging: LoggingArgs = field(default_factory=LoggingArgs)
    distributed: DistributedArgs = field(default_factory=DistributedArgs)
    env: EnvironmentArgs = field(default_factory=EnvironmentArgs)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class VideoTextDataset(Dataset):
    """
    Video-text contrastive dataset.

    Reads a JSONL file where each line is a JSON object in conversations format:
        {
            "video":   "clip.mp4",
            "video_id": "Abuse001_x264_000",       # optional
            "source":  "VATEX",                     # optional
            "type":    "captioning",                # optional
            "instruction": "...",                   # optional, unused
            "conversations": [
                {"from": "human",     "value": "..."},
                {"from": "assistant", "value": "caption used as the text pair"}
            ]
        }
    The caption is the value of the last "assistant" turn in conversations.
    The video filename is joined with ``video_root`` (if provided) to form the
    full path.

    Each sample is decoded on the fly: frames are loaded, uniformly sampled
    to exactly ``num_frames``, resized to ``image_size x image_size``, and
    normalised to [-1, 1].  Short videos are zero-padded (black frames) to
    ensure a constant temporal length suitable for batching.

    Returns:
        frames : Float tensor (num_frames, 3, H, W)
        tokens : Long  tensor (context_length,)
    """

    def __init__(
        self,
        jsonl_path: str,
        video_transform: VideoTransform,
        tokenizer,
        num_frames: int = 8,
        sampling_fps: int = 1,
        video_root: str = "",
    ):
        super().__init__()
        self.video_transform = video_transform
        self.tokenizer = tokenizer
        self.num_frames = num_frames
        self.sampling_fps = sampling_fps
        self.video_root = video_root

        self.samples: List[Dict] = []
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    self.samples.append(json.loads(line))

        logger.info(f"Loaded {len(self.samples):,} samples from {jsonl_path}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        item = self.samples[idx]

        # Resolve video path
        video_file = item["video"]
        video_path = (
            str(Path(self.video_root) / video_file)
            if self.video_root
            else video_file
        )

        # Extract caption from the last assistant turn in conversations
        caption = ""
        for turn in reversed(item.get("conversations", [])):
            if turn.get("from") == "assistant":
                caption = turn.get("value", "")
                break

        frames = self._load_frames(video_path, start=None, end=None)
        tokens = self.tokenizer(caption)   # (context_length,)
        return frames, tokens

    def _load_frames(
        self, video_path: str, start: Optional[float], end: Optional[float]
    ) -> torch.Tensor:
        """
        Load and preprocess video frames, padding to exactly ``num_frames``.

        Returns: Float tensor (num_frames, 3, H, W)
        """
        video_info = (video_path, self.num_frames, start, end, None)
        try:
            frames, _ = self.video_transform(
                video_info, sampling_fps=self.sampling_fps
            )
            # frames: (N, 3, H, W) where N <= num_frames
        except Exception as e:
            logger.warning(f"Failed to load {video_path}: {e}")
            # Return a zero tensor (black video) so that training can continue
            h = w = self.video_transform.size
            return torch.zeros(self.num_frames, 3, h, w)

        # Pad short clips with black frames so every sample has the same shape
        if frames.shape[0] < self.num_frames:
            pad_n = self.num_frames - frames.shape[0]
            pad = torch.zeros(pad_n, *frames.shape[1:], dtype=frames.dtype)
            frames = torch.cat([frames, pad], dim=0)

        # Truncate in the unlikely case the transform returned more frames
        return frames[: self.num_frames]


def _collate_fn(batch: List[Tuple]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Stack (frames, tokens) pairs into batched tensors."""
    frames_list, tokens_list = zip(*batch)
    return torch.stack(frames_list, dim=0), torch.stack(tokens_list, dim=0)


def build_dataloader(
    args: DataArgs,
    tokenizer,
    split: str = "train",
) -> Optional[DataLoader]:
    jsonl_path = args.train_data if split == "train" else args.val_data
    if not jsonl_path:
        return None

    video_transform = VideoTransform(size=args.image_size)
    dataset = VideoTextDataset(
        jsonl_path=jsonl_path,
        video_transform=video_transform,
        tokenizer=tokenizer,
        num_frames=args.num_frames,
        sampling_fps=args.sampling_fps,
        video_root=args.video_root,
    )

    is_train = split == "train"
    sampler = (
        DistributedSampler(
            dataset,
            num_replicas=get_world_size(),
            rank=get_global_rank(),
            shuffle=is_train,
            seed=42,
        )
        if dist.is_initialized()
        else None
    )

    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=(sampler is None and is_train),
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        drop_last=is_train,
        collate_fn=_collate_fn,
    )


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def contrastive_loss(
    video_feats: torch.Tensor,
    text_feats: torch.Tensor,
    logit_scale: torch.Tensor,
) -> torch.Tensor:
    """
    Symmetric InfoNCE contrastive loss (identical to CLIP training objective).

    In a distributed setting each rank contributes its local batch; features
    are gathered across all ranks so that each sample is contrasted against
    all other samples in the global batch.  The gather operation uses
    ``all_gather`` (no gradient flows through gathered neighbours) which is
    the standard CLIP training trick.

    Args:
        video_feats  : (B, D) L2-normalised video embeddings from this rank
        text_feats   : (B, D) L2-normalised text  embeddings from this rank
        logit_scale  : scalar exp(log_scale) temperature

    Returns:
        Scalar loss tensor.
    """
    if dist.is_initialized():
        all_video = _all_gather_with_grad(video_feats)
        all_text = _all_gather_with_grad(text_feats)
    else:
        all_video, all_text = video_feats, text_feats

    logits_v2t = logit_scale * all_video @ all_text.T   # (N, N)
    logits_t2v = logits_v2t.T                           # (N, N)

    N = logits_v2t.shape[0]
    labels = torch.arange(N, device=logits_v2t.device)

    loss = 0.5 * (
        F.cross_entropy(logits_v2t, labels) +
        F.cross_entropy(logits_t2v, labels)
    )
    return loss


class _AllGather(torch.autograd.Function):
    """
    All-gather with gradient support for the local rank's slice.

    Only the local rank's shard receives gradients; gathered neighbours are
    treated as constants (identical behaviour to the original CLIP codebase).
    """

    @staticmethod
    def forward(ctx, tensor: torch.Tensor) -> torch.Tensor:
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        ctx.rank = rank
        ctx.batch_size = tensor.shape[0]

        gathered = [torch.empty_like(tensor) for _ in range(world_size)]
        dist.all_gather(gathered, tensor.contiguous())
        return torch.cat(gathered, dim=0)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        # Return gradient only for this rank's slice
        start = ctx.rank * ctx.batch_size
        return grad_output[start : start + ctx.batch_size].contiguous()


def _all_gather_with_grad(t: torch.Tensor) -> torch.Tensor:
    return _AllGather.apply(t)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_model(args: ModelArgs) -> CLIP:
    """
    Instantiate CLIP with a TemporalTransformer and optionally freeze the
    pretrained spatial towers so that only the temporal head is trained.
    """
    model = CLIP.from_config(
        name=args.pe_config,
        pretrained=args.pretrained,
        checkpoint_path=args.checkpoint_path,
    )

    # Replace the default temporal encoder if non-standard depth/heads requested
    if (
        model.temporal_encoder is None
        or model.temporal_encoder.resblocks.__len__() != args.num_temporal_layers
    ):
        from core.vision_encoder.pe import TemporalTransformer
        from core.vision_encoder.config import PE_VISION_CONFIG

        output_dim = PE_VISION_CONFIG[args.pe_config].output_dim
        if output_dim is not None:
            model.temporal_encoder = TemporalTransformer(
                width=output_dim,
                layers=args.num_temporal_layers,
                heads=args.num_temporal_heads,
            )

    if args.freeze_spatial:
        # Freeze the pretrained spatial image encoder
        for param in model.visual.parameters():
            param.requires_grad_(False)
        # Freeze the pretrained text encoder (all TextTransformer params except logit_scale)
        for name, param in model.named_parameters():
            if any(
                name.startswith(prefix)
                for prefix in (
                    "transformer",       # TextTransformer blocks
                    "token_embedding",   # Text token embeddings
                    "ln_final",          # Text final layer norm
                    "text_projection",   # Text projection head
                    "positional_embedding",  # Text positional embeddings
                )
            ):
                param.requires_grad_(False)

    if args.freeze_logit_scale:
        model.logit_scale.requires_grad_(False)

    return model


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """
    Compute video-text retrieval R@1 on the full validation set.

    Features from all validation batches are collected and the similarity
    matrix is computed globally, giving exact (not approximate) recall.

    Returns a dict with keys "val/r1_v2t" and "val/r1_t2v".
    """
    raw_model = model.module if isinstance(model, DDP) else model
    raw_model.eval()

    all_video_feats, all_text_feats = [], []
    for videos, tokens in loader:
        videos = videos.to(device, non_blocking=True)  # (B, N, 3, H, W)
        tokens = tokens.to(device, non_blocking=True)  # (B, ctx)

        v_feats = raw_model.encode_video(videos, normalize=True)  # (B, D)
        t_feats = raw_model.encode_text(tokens, normalize=True)   # (B, D)

        all_video_feats.append(v_feats.cpu())
        all_text_feats.append(t_feats.cpu())

    video_feats = torch.cat(all_video_feats, dim=0)  # (N, D)
    text_feats = torch.cat(all_text_feats, dim=0)    # (N, D)

    sim = video_feats @ text_feats.T   # (N, N)
    n = sim.shape[0]
    gt = torch.arange(n)

    r1_v2t = (sim.argmax(dim=1) == gt).float().mean().item()
    r1_t2v = (sim.argmax(dim=0) == gt).float().mean().item()

    raw_model.train()
    return {"val/r1_v2t": r1_v2t, "val/r1_t2v": r1_t2v}


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    step: int,
    path: Path,
) -> None:
    raw_model = model.module if isinstance(model, DDP) else model
    torch.save(
        {
            "model": raw_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "step": step,
        },
        path,
    )
    logger.info(f"Checkpoint saved → {path}")


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
) -> int:
    """Load checkpoint and return the step to resume from."""
    logger.info(f"Resuming from checkpoint {path}")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    raw_model = model.module if isinstance(model, DDP) else model
    missing, unexpected = raw_model.load_state_dict(ckpt["model"], strict=False)
    if missing:
        logger.warning(f"Missing keys when loading checkpoint: {missing}")
    if unexpected:
        logger.warning(f"Unexpected keys when loading checkpoint: {unexpected}")
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    return ckpt["step"] + 1


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args: TrainArgs) -> None:
    # ------------------------------------------------------------------
    # Device
    # ------------------------------------------------------------------
    if dist.is_initialized():
        device = torch.device(f"cuda:{get_local_rank()}")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # ------------------------------------------------------------------
    # Output directory & config dump
    # ------------------------------------------------------------------
    if get_is_master():
        Path(args.dump_dir).mkdir(parents=True, exist_ok=True)
        OmegaConf.save(
            OmegaConf.structured(args),
            str(Path(args.dump_dir) / "config.yaml"),
        )

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model = build_model(args.model).to(device)

    if dist.is_initialized():
        model = DDP(model, device_ids=[get_local_rank()], find_unused_parameters=False)

    if get_is_master():
        log_model_params(model.module if isinstance(model, DDP) else model)

    raw_model = model.module if isinstance(model, DDP) else model

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    tokenizer = get_text_tokenizer(raw_model.context_length)

    train_loader = build_dataloader(args.data, tokenizer, split="train")
    val_loader = (
        build_dataloader(args.data, tokenizer, split="val")
        if args.data.val_data
        else None
    )

    assert train_loader is not None, (
        "No training data — set data.train_data to your JSONL file path."
    )

    steps_per_epoch = len(train_loader)

    # ------------------------------------------------------------------
    # Optimizer & LR scheduler
    # ------------------------------------------------------------------
    optimizer, scheduler = build_optimizer(raw_model, args.optim, args.max_steps)

    # ------------------------------------------------------------------
    # Resume from checkpoint (latest checkpoint.pt in dump_dir)
    # ------------------------------------------------------------------
    start_step = 0
    ckpt_path = Path(args.dump_dir) / "checkpoint.pt"
    if ckpt_path.exists():
        start_step = load_checkpoint(ckpt_path, model, optimizer, scheduler, device)
        logger.info(f"Resumed training at step {start_step}")

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    metrics_path = Path(args.dump_dir) / "metrics.jsonl"
    with MetricLogger(metrics_path, args=args) as metric_logger:
        model.train()
        step = start_step
        running_loss = 0.0
        optimizer.zero_grad()

        while step < args.max_steps:
            # Advance the DistributedSampler epoch counter for proper shuffling
            if isinstance(getattr(train_loader, "sampler", None), DistributedSampler):
                train_loader.sampler.set_epoch(step // steps_per_epoch)

            for videos, tokens in train_loader:
                if step >= args.max_steps:
                    break

                # videos : (B, N, 3, H, W)
                # tokens : (B, context_length)
                videos = videos.to(device, non_blocking=True)
                tokens = tokens.to(device, non_blocking=True)

                # Forward pass
                video_feats = raw_model.encode_video(videos, normalize=True)   # (B, D)
                text_feats = raw_model.encode_text(tokens, normalize=True)     # (B, D)

                loss = contrastive_loss(
                    video_feats, text_feats, raw_model.logit_scale.exp()
                )

                # Scale loss for gradient accumulation
                (loss / args.grad_accum_steps).backward()
                running_loss += loss.item()

                # Optimizer step (every grad_accum_steps)
                if (step + 1) % args.grad_accum_steps == 0:
                    if args.max_grad_norm > 0:
                        nn.utils.clip_grad_norm_(
                            model.parameters(), args.max_grad_norm
                        )
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                # ----------------------------------------------------------
                # Logging
                # ----------------------------------------------------------
                if get_is_master() and step % args.log_freq == 0:
                    avg_loss = running_loss / max(step - start_step + 1, 1)
                    running_loss = 0.0
                    lr = scheduler.get_last_lr()[0]
                    log_scale = raw_model.logit_scale.item()

                    metrics = {
                        "global_step": step,
                        "train/loss": avg_loss,
                        "train/lr": lr,
                        "train/logit_scale": log_scale,
                    }
                    metric_logger.log(metrics)
                    logger.info(
                        f"step={step:6d}  loss={avg_loss:.4f}  "
                        f"lr={lr:.2e}  logit_scale={log_scale:.3f}"
                    )

                # ----------------------------------------------------------
                # Validation
                # ----------------------------------------------------------
                if (
                    val_loader is not None
                    and args.eval_freq > 0
                    and step > 0
                    and step % args.eval_freq == 0
                ):
                    val_metrics = evaluate(model, val_loader, device)
                    val_metrics["global_step"] = step
                    if get_is_master():
                        metric_logger.log(val_metrics)
                        logger.info(
                            f"val  R@1  v→t={val_metrics['val/r1_v2t']:.4f}  "
                            f"t→v={val_metrics['val/r1_t2v']:.4f}"
                        )

                # ----------------------------------------------------------
                # Checkpoint
                # ----------------------------------------------------------
                if get_is_master() and step > 0 and step % args.save_freq == 0:
                    save_checkpoint(
                        model, optimizer, scheduler, step,
                        Path(args.dump_dir) / f"checkpoint_{step:07d}.pt",
                    )
                    # Also keep a rolling "latest" checkpoint for easy resuming
                    save_checkpoint(
                        model, optimizer, scheduler, step, ckpt_path
                    )

                step += 1

        # Final checkpoint
        if get_is_master():
            save_checkpoint(
                model, optimizer, scheduler, step - 1,
                Path(args.dump_dir) / f"checkpoint_final.pt",
            )

    logger.info("Training complete.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Config priority (highest → lowest):
        1. CLI key=value overrides
        2. --config path/to/config.yaml
        3. dataclass defaults
    """
    # Parse CLI
    cli_args = OmegaConf.from_cli()

    if "config" in cli_args:
        config_file = cli_args.pop("config")
        base_cfg = OmegaConf.load(config_file)
    else:
        base_cfg = OmegaConf.structured(TrainArgs())

    cfg = OmegaConf.merge(base_cfg, cli_args)
    args: TrainArgs = OmegaConf.to_object(cfg)

    # Distributed setup
    if get_world_size() > 1 or os.environ.get("LOCAL_RANK"):
        setup_env(args.env)
        setup_torch_distributed(args.distributed)

    # Logging setup
    logging.basicConfig(
        level=getattr(logging, args.logging.level, logging.INFO),
        format="%(asctime)s [rank %(process)d] [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )

    # Reproducibility
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if get_is_master():
        logger.info(f"Config:\n{OmegaConf.to_yaml(OmegaConf.structured(args))}")

    train(args)


if __name__ == "__main__":
    main()
