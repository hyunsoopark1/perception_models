#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
PLM Feature Extraction for Video-Text Cosine Similarity.

Extracts video and text features from Perception Language Model (PLM) for
computing cosine similarity. Unlike the Perception Encoder (PE/CLIP) which
uses a contrastive dual-encoder architecture, PLM processes both modalities
through a shared LLM backbone, producing embeddings in a unified hidden-state
space.

Architecture:
  Video: frames -> VisionTransformer -> MLPProjector -+
                                                      +-> LLM layers -> RMSNorm -> pool -> normalize
  Text:  tokens -> tok_embeddings --------------------+

Pooling strategy (critical for quality):
  - Video: mean-pool over image token positions only (excludes chat template
    overhead), capturing the visual content as encoded by the LLM.
  - Text: raw text encoding (no chat template), mean-pool over content tokens
    (excluding BOS), so the embedding reflects the actual text semantics.

Both paths produce L2-normalized embeddings of the same dimension (e.g., 3072
for PLM-3B), enabling direct cosine similarity computation.

Usage:
    # Compare a video against text queries
    python extract_plm_features.py \
        --video path/to/video.mp4 \
        --texts "a person riding a bike" "a dog playing fetch" \
        --ckpt facebook/Perception-LM-3B

    # Multiple videos vs multiple texts
    python extract_plm_features.py \
        --video_dir path/to/videos/ \
        --texts "cooking" "sports" "nature" \
        --output results.json

    # Text queries from a file (one per line)
    python extract_plm_features.py \
        --video path/to/video.mp4 \
        --text_file queries.txt

API Usage:
    from extract_plm_features import (
        encode_video, encode_text, encode_videos, encode_texts,
        compute_similarity_matrix,
    )
    from apps.plm.generate import load_consolidated_model_and_tokenizer

    model, tokenizer, config = load_consolidated_model_and_tokenizer(
        "facebook/Perception-LM-3B"
    )

    video_emb = encode_video(model, tokenizer, "video.mp4")
    text_emb = encode_text(model, tokenizer, "a person walking")
    similarity = (video_emb @ text_emb).item()
"""

import argparse
import json
import logging
import os
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

from apps.plm.generate import load_consolidated_model_and_tokenizer
from apps.plm.tokenizer import PLMTokenizer
from apps.plm.transformer import LMTransformer, create_causal_mask
from core.transforms.video_transform import get_video_transform

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv"}


@torch.inference_mode()
def extract_hidden_states(
    model: LMTransformer,
    token_values: torch.Tensor,
    images: Optional[torch.Tensor] = None,
    image_pos_index: Optional[torch.Tensor] = None,
    num_chunks: Optional[List[int]] = None,
    media_type: Optional[List[str]] = None,
) -> torch.Tensor:
    """Run the PLM forward pass and return hidden states before the LM head.

    This replicates the LMTransformer.forward() pipeline but returns the
    RMSNorm-normalized hidden states instead of logits, making them suitable
    as feature embeddings for similarity computation.

    Args:
        model: The PLM LMTransformer model.
        token_values: Input token IDs, shape (1, seqlen).
        images: Optional video/image frames, shape (num_frames, C, H, W).
        image_pos_index: Tensor indicating where image tokens are placed,
            shape (1, seqlen). Positions with values >= 0 contain image tokens.
        num_chunks: Number of image/frame chunks per sample.
        media_type: Media type identifier for each sample.

    Returns:
        Hidden states tensor of shape (1, seqlen, dim).
    """
    num_chunks = num_chunks or [1]
    media_type = media_type or ["video"]

    _, seqlen = token_values.shape

    # Step 1: Token embeddings
    h = model.tok_embeddings(token_values)

    # Step 2: Stitch vision features into the token sequence
    if images is not None and image_pos_index is not None:
        h_img = model.vision_model(images, strip_cls_token=True)
        h_img = model.vision_projector(h_img)
        h = model.stitch_images_into_text(
            h,
            h_img,
            image_pos_index,
            num_chunks=num_chunks,
            media_type=media_type,
        )

    # Step 3: Causal attention mask
    mask = create_causal_mask(seqlen, "sdpa", model.sliding_window)

    # Step 4: Run through all transformer layers
    freq_cis = model.rope_embeddings(seqlen=model.max_seqlen)
    for layer in model.layers:
        h = layer(h, freq_cis, mask=mask, attn_impl="sdpa")

    # Step 5: Final RMS normalization
    h = model.norm(h)

    return h


@torch.inference_mode()
def encode_video(
    model: LMTransformer,
    tokenizer: PLMTokenizer,
    video_path: str,
    num_frames: int = 16,
    prompt: str = "",
    pool: str = "mean",
) -> torch.Tensor:
    """Extract a normalized video embedding from PLM.

    Processes video frames through the full PLM pipeline:
    VisionTransformer -> MLPProjector -> LLM layers -> embedding.

    The default pooling ("mean") averages hidden states at image token
    positions only, producing an embedding that captures the visual content
    without noise from the surrounding chat template tokens.

    Args:
        model: The PLM model.
        tokenizer: The PLM tokenizer.
        video_path: Path to the video file.
        num_frames: Number of frames to uniformly sample.
        prompt: Optional text prompt appended after image tokens in the
            user message (e.g., "Describe this video."). Empty by default.
        pool: Pooling strategy:
            - "mean" (default): mean over image token positions only.
              Best for retrieval — captures visual content, ignores
              chat template overhead.
            - "mean_all": mean over all sequence positions.
            - "last": last-token hidden state (includes chat template
              context; less discriminative for retrieval).

    Returns:
        L2-normalized embedding tensor of shape (dim,).
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    # Load and transform video frames
    transform = get_video_transform(image_res=model.vision_model.image_size)
    video_info = (video_path, num_frames, None, None, None)
    frames, _ = transform(video_info)  # (N, C, H, W)

    # Build tokenized prompt with image token placeholders
    text_ids, image_pos = tokenizer._tokenize_for_generation(prompt, frames)
    token_values = torch.tensor([text_ids], dtype=torch.long, device="cuda")

    # Build image_pos_index: maps text positions to sequential image token indices
    image_pos_index = torch.full(
        token_values.shape, -1, dtype=torch.int, device="cuda"
    )
    image_pos_index[0, image_pos] = torch.arange(
        len(image_pos), dtype=torch.int, device="cuda"
    )

    # Move frames to model device and dtype
    param = next(model.parameters())
    images = frames.to(device=param.device, dtype=param.dtype)

    # Extract hidden states from the full PLM pipeline
    h = extract_hidden_states(
        model,
        token_values,
        images=images,
        image_pos_index=image_pos_index,
        num_chunks=[frames.size(0)],
        media_type=["video"],
    )

    # Pool hidden states to a single vector
    if pool == "mean":
        # Mean over image token positions only — captures visual content
        # without noise from system prompt, headers, and structural tokens.
        img_positions = torch.tensor(image_pos, device=h.device)
        embedding = h[0, img_positions].mean(dim=0)
    elif pool == "mean_all":
        embedding = h[0].mean(dim=0)
    elif pool == "last":
        embedding = h[0, -1, :]
    else:
        raise ValueError(f"Unknown pool type: {pool}")

    return F.normalize(embedding, dim=-1)


@torch.inference_mode()
def encode_text(
    model: LMTransformer,
    tokenizer: PLMTokenizer,
    text: str,
    pool: str = "mean",
) -> torch.Tensor:
    """Extract a normalized text embedding from PLM.

    Encodes text directly through the LLM (BOS + raw text tokens, without
    the chat template) so the embedding reflects the text content rather
    than being dominated by shared structural tokens.

    The default pooling ("mean") averages over content token positions
    (excluding BOS), producing a content-focused embedding comparable
    to the image-token-pooled video embedding.

    Args:
        model: The PLM model.
        tokenizer: The PLM tokenizer.
        text: Input text string.
        pool: Pooling strategy:
            - "mean" (default): mean over content token positions
              (excludes BOS). Best for retrieval.
            - "mean_all": mean over all positions including BOS.
            - "last": last-token hidden state.

    Returns:
        L2-normalized embedding tensor of shape (dim,).
    """
    # Encode raw text without chat template — avoids the system prompt and
    # header tokens that would dominate the representation and make all
    # text embeddings look similar regardless of content.
    text_ids = tokenizer.encode(text, add_bos=True, add_eos=False)
    token_values = torch.tensor([text_ids], dtype=torch.long, device="cuda")

    h = extract_hidden_states(model, token_values)

    if pool == "mean":
        # Skip BOS (position 0), pool over content tokens only
        embedding = h[0, 1:].mean(dim=0)
    elif pool == "mean_all":
        embedding = h[0].mean(dim=0)
    elif pool == "last":
        embedding = h[0, -1, :]
    else:
        raise ValueError(f"Unknown pool type: {pool}")

    return F.normalize(embedding, dim=-1)


@torch.inference_mode()
def encode_videos(
    model: LMTransformer,
    tokenizer: PLMTokenizer,
    video_paths: List[str],
    num_frames: int = 16,
    prompt: str = "",
    pool: str = "mean",
) -> torch.Tensor:
    """Encode multiple videos into a stacked embedding tensor.

    Args:
        model: The PLM model.
        tokenizer: The PLM tokenizer.
        video_paths: List of video file paths.
        num_frames: Number of frames per video.
        prompt: Optional prompt for video encoding.
        pool: Pooling strategy.

    Returns:
        Stacked L2-normalized embeddings, shape (num_videos, dim).
    """
    embeddings = []
    for path in video_paths:
        emb = encode_video(model, tokenizer, path, num_frames, prompt, pool)
        embeddings.append(emb)
    return torch.stack(embeddings)


@torch.inference_mode()
def encode_texts(
    model: LMTransformer,
    tokenizer: PLMTokenizer,
    texts: List[str],
    pool: str = "mean",
) -> torch.Tensor:
    """Encode multiple texts into a stacked embedding tensor.

    Args:
        model: The PLM model.
        tokenizer: The PLM tokenizer.
        texts: List of text strings.
        pool: Pooling strategy.

    Returns:
        Stacked L2-normalized embeddings, shape (num_texts, dim).
    """
    embeddings = []
    for text in texts:
        emb = encode_text(model, tokenizer, text, pool)
        embeddings.append(emb)
    return torch.stack(embeddings)


def compute_similarity_matrix(
    video_embeddings: torch.Tensor,
    text_embeddings: torch.Tensor,
) -> torch.Tensor:
    """Compute cosine similarity between video and text embeddings.

    Since both embedding sets are L2-normalized, this reduces to a
    matrix dot product.

    Args:
        video_embeddings: Shape (num_videos, dim).
        text_embeddings: Shape (num_texts, dim).

    Returns:
        Similarity matrix of shape (num_videos, num_texts), with values
        in [-1, 1].
    """
    return video_embeddings @ text_embeddings.T


def collect_videos(video_dir: str) -> List[str]:
    """Recursively collect all supported video files from a directory."""
    videos = []
    for root, _, files in os.walk(video_dir):
        for f in sorted(files):
            if Path(f).suffix.lower() in SUPPORTED_VIDEO_EXTENSIONS:
                videos.append(os.path.join(root, f))
    return videos


def main():
    parser = argparse.ArgumentParser(
        description="Extract video and text features from PLM for cosine similarity.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --video clip.mp4 --texts "a cat" "a dog"
  %(prog)s --video_dir ./videos/ --texts "cooking" "sports" --output results.json
  %(prog)s --video clip.mp4 --text_file queries.txt --pool last
        """,
    )

    # Video input
    video_group = parser.add_mutually_exclusive_group(required=True)
    video_group.add_argument(
        "--video", type=str, help="Path to a single video file."
    )
    video_group.add_argument(
        "--video_dir", type=str, help="Directory containing video files."
    )

    # Text input
    text_group = parser.add_mutually_exclusive_group(required=True)
    text_group.add_argument(
        "--texts", nargs="+", type=str, help="One or more text queries."
    )
    text_group.add_argument(
        "--text_file",
        type=str,
        help="Path to a text file with one query per line.",
    )

    # Model config
    parser.add_argument(
        "--ckpt",
        type=str,
        default="facebook/Perception-LM-3B",
        help="Model checkpoint or HuggingFace ID (default: facebook/Perception-LM-3B).",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=16,
        help="Number of frames to sample from each video (default: 16).",
    )
    parser.add_argument(
        "--pool",
        type=str,
        default="mean",
        choices=["mean", "mean_all", "last"],
        help=(
            "Pooling strategy for hidden states (default: mean). "
            "'mean' pools over content tokens only (image positions for video, "
            "text tokens for text). 'mean_all' pools over all positions. "
            "'last' uses the last token's hidden state."
        ),
    )
    parser.add_argument(
        "--video_prompt",
        type=str,
        default="",
        help="Optional prompt appended after image tokens during video encoding.",
    )

    # Output
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Save results to a JSON file.",
    )

    args = parser.parse_args()

    # Load model
    logger.info(f"Loading PLM from: {args.ckpt}")
    model, tokenizer, config = load_consolidated_model_and_tokenizer(args.ckpt)

    # Collect video paths
    if args.video:
        video_paths = [args.video]
    else:
        video_paths = collect_videos(args.video_dir)
        if not video_paths:
            logger.error(f"No supported video files found in {args.video_dir}")
            return
        logger.info(f"Found {len(video_paths)} video(s)")

    # Collect text queries
    if args.texts:
        texts = args.texts
    else:
        with open(args.text_file) as f:
            texts = [line.strip() for line in f if line.strip()]
        logger.info(f"Loaded {len(texts)} text queries from {args.text_file}")

    # Encode videos
    logger.info(f"Encoding {len(video_paths)} video(s)...")
    video_embeddings = encode_videos(
        model,
        tokenizer,
        video_paths,
        num_frames=args.num_frames,
        prompt=args.video_prompt,
        pool=args.pool,
    )

    # Encode texts
    logger.info(f"Encoding {len(texts)} text(s)...")
    text_embeddings = encode_texts(model, tokenizer, texts, pool=args.pool)

    # Compute similarity matrix
    sim_matrix = compute_similarity_matrix(video_embeddings, text_embeddings)

    # Display results
    print(f"\n{'=' * 70}")
    print("Video-Text Cosine Similarity (PLM)")
    print(f"{'=' * 70}")
    print(f"Model: {args.ckpt}")
    print(f"Pooling: {args.pool}")
    print(f"Embedding dim: {video_embeddings.shape[-1]}")
    print(f"Videos: {len(video_paths)}, Texts: {len(texts)}")
    print(f"{'=' * 70}\n")

    results = []
    for i, vpath in enumerate(video_paths):
        vname = os.path.basename(vpath)
        print(f"Video: {vname}")
        print(f"{'-' * 50}")
        video_scores = []
        for j, text in enumerate(texts):
            score = sim_matrix[i, j].item()
            print(f"  [{score:+.4f}] {text}")
            video_scores.append({"text": text, "similarity": round(score, 6)})
        print()
        results.append({"video": vpath, "scores": video_scores})

    # Save results if requested
    if args.output:
        output_data = {
            "model": args.ckpt,
            "pooling": args.pool,
            "num_frames": args.num_frames,
            "embedding_dim": video_embeddings.shape[-1],
            "results": results,
        }
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        logger.info(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
