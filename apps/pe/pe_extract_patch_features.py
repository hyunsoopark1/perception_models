"""
PE Patch Feature Extractor  —  offline preprocessing step.

Runs the frozen PE visual encoder over a video file (or a directory of frames)
and saves per-frame patch tokens to a single .pt file.  This is the expensive
step; run it once, then use pe_track_query.py for fast repeated queries with
any number of bounding-box tracks — no image encoder reload required.

Saved file layout
-----------------
    patch_tokens  : Tensor[T, N, D]   — frozen PE spatial patch tokens (float32)
    proj          : Tensor[D, E] | None — visual projection matrix (D → CLIP dim)
    frame_keys    : list[str]          — frame identifiers (see below)
    model_name    : str
    image_size    : int                — encoder input resolution (e.g. 448)
    patch_size    : int                — patch side in pixels  (e.g. 14)
    width         : int  (D)          — patch token dimension

    # video source only:
    source_video  : str               — absolute path to the input video
    video_fps     : float             — original video frame rate
    sample_fps    : float             — frame rate at which frames were sampled
    frame_indices : list[int]         — original frame indices in the video

    frame_keys for video  : "000000", "000001", …  (original frame index, zero-padded)
    frame_keys for frames : filename of each image (e.g. "frame_000.jpg")

Usage
-----
    # From a video file (all frames):
    python apps/pe/pe_extract_patch_features.py \\
        --video  /path/to/video.mp4 \\
        --out    patch_features.pt \\
        --model  PE-Core-G14-448

    # Sample at 5 fps from a video:
    python apps/pe/pe_extract_patch_features.py \\
        --video  /path/to/video.mp4 \\
        --out    patch_features.pt \\
        --fps    5

    # Process only frames 100–500 of a video:
    python apps/pe/pe_extract_patch_features.py \\
        --video        /path/to/video.mp4 \\
        --out          patch_features.pt \\
        --start-frame  100 \\
        --end-frame    500

    # From a directory of frames (original behaviour):
    python apps/pe/pe_extract_patch_features.py \\
        --frame-dir /path/to/frames \\
        --out       patch_features.pt

    # Halve disk usage with float16 tokens:
    python apps/pe/pe_extract_patch_features.py \\
        --video /path/to/video.mp4 \\
        --out   patch_features.pt \\
        --half
"""

import argparse
import sys
from pathlib import Path
from typing import Iterator, Tuple

import torch
from PIL import Image as PILImage

# Image extensions recognised when using --frame-dir
_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ---------------------------------------------------------------------------
# Frame iterators
# ---------------------------------------------------------------------------

def _iter_video_frames(
    video_path: str,
    fps: float | None,
    start_frame: int,
    end_frame: int | None,
) -> Iterator[Tuple[int, PILImage.Image]]:
    """
    Yield (original_frame_index, PIL RGB image) from a video file.

    Parameters
    ----------
    fps         : target sampling rate; None = every frame
    start_frame : first frame index to include (0-based)
    end_frame   : last frame index (exclusive); None = until EOF
    """
    try:
        import cv2
    except ImportError:
        sys.exit(
            "opencv-python is required for video input.\n"
            "Install with:  pip install opencv-python-headless"
        )

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        sys.exit(f"Cannot open video: {video_path}")

    video_fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total       = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sample_fps  = fps if fps is not None else video_fps
    stride      = max(1, round(video_fps / sample_fps))

    print(f"  video fps={video_fps:.3f}  total_frames={total}"
          f"  sample_fps={sample_fps:.3f}  stride={stride}")

    # Seek to start_frame efficiently
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frame_idx = start_frame
    try:
        while True:
            ret, bgr = cap.read()
            if not ret:
                break
            if end_frame is not None and frame_idx >= end_frame:
                break

            if (frame_idx - start_frame) % stride == 0:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                yield frame_idx, PILImage.fromarray(rgb)

            frame_idx += 1
    finally:
        cap.release()


def _iter_dir_frames(
    frame_dir: str,
) -> Iterator[Tuple[str, PILImage.Image]]:
    """Yield (filename, PIL RGB image) for every image in a directory (sorted)."""
    paths = sorted(
        p for p in Path(frame_dir).iterdir() if p.suffix.lower() in _IMG_EXTS
    )
    if not paths:
        sys.exit(f"No images found in {frame_dir}  (extensions: {_IMG_EXTS})")
    for p in paths:
        yield p.name, PILImage.open(p).convert("RGB")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(
        description="Extract PE patch tokens from a video file or frame directory.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--video", metavar="PATH",
                     help="Input video file (.mp4, .avi, .mov, …).")
    src.add_argument("--frame-dir", metavar="DIR",
                     help="Directory of frame images (sorted alphabetically).")

    p.add_argument("--out", required=True, metavar="PATH",
                   help="Output .pt file.")
    p.add_argument("--model", default="PE-Core-G14-448",
                   choices=["PE-Core-G14-448", "PE-Core-L14-336",
                            "PE-Core-B16-224", "PE-Core-S16-384", "PE-Core-T16-384"],
                   help="PE model variant (default: PE-Core-G14-448).")
    p.add_argument("--batch-size", type=int, default=8, metavar="N",
                   help="Frames per GPU batch (default: 8).")
    p.add_argument("--checkpoint", metavar="PATH",
                   help="Local .pt path for the PE encoder (overrides HF download).")
    p.add_argument("--no-pretrained", action="store_true",
                   help="Skip loading pretrained weights (smoke test).")
    p.add_argument("--half", action="store_true",
                   help="Save patch tokens as float16 (~2× smaller file).")

    # Video-only options
    p.add_argument("--fps", type=float, default=None, metavar="N",
                   help="[video] Sample at this frame rate; default = every frame.")
    p.add_argument("--start-frame", type=int, default=0, metavar="N",
                   help="[video] First frame index to process (default: 0).")
    p.add_argument("--end-frame", type=int, default=None, metavar="N",
                   help="[video] Last frame index, exclusive (default: until EOF).")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = _parse_args()

    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pretrained = not args.no_pretrained

    # ------------------------------------------------------------------
    # Load PE visual encoder
    # ------------------------------------------------------------------
    from core.vision_encoder import transforms as pe_transforms
    from core.vision_encoder.pe import CLIP

    print(f"Loading {args.model} (pretrained={pretrained}) on {device} …")
    pe_model = CLIP.from_config(
        args.model,
        pretrained=pretrained,
        checkpoint_path=args.checkpoint,
    ).to(device).eval()

    visual     = pe_model.visual
    preprocess = pe_transforms.get_image_transform(visual.image_size)

    # ------------------------------------------------------------------
    # Build frame iterator
    # ------------------------------------------------------------------
    is_video = args.video is not None

    if is_video:
        print(f"Source : video  {args.video}")
        frame_iter = _iter_video_frames(
            args.video,
            fps=args.fps,
            start_frame=args.start_frame,
            end_frame=args.end_frame,
        )
    else:
        print(f"Source : frames  {args.frame_dir}")
        frame_iter = _iter_dir_frames(args.frame_dir)

    # ------------------------------------------------------------------
    # Extract patch tokens in batches
    # ------------------------------------------------------------------
    all_tokens:       list[torch.Tensor] = []
    frame_keys:       list[str]          = []
    frame_indices:    list[int]          = []   # video only
    video_fps_value:  float              = 0.0
    sample_fps_value: float              = 0.0

    buf_imgs: list[torch.Tensor] = []
    buf_keys: list[str]          = []

    def _flush_buf():
        if not buf_imgs:
            return
        imgs   = torch.stack(buf_imgs).to(device)
        tokens = visual.forward_features(imgs, norm=True, strip_cls_token=True)
        all_tokens.append(tokens.cpu())
        frame_keys.extend(buf_keys)
        buf_imgs.clear()
        buf_keys.clear()
        print(f"\r  {len(frame_keys)} frames encoded", end="", flush=True)

    with torch.no_grad():
        for key, pil in frame_iter:
            if is_video:
                # key is the original integer frame index
                frame_indices.append(int(key))
                buf_keys.append(f"{int(key):06d}")
            else:
                buf_keys.append(str(key))   # filename

            buf_imgs.append(preprocess(pil))

            if len(buf_imgs) == args.batch_size:
                _flush_buf()

        _flush_buf()

    if not frame_keys:
        sys.exit("No frames were read — check your input path and frame range.")

    T = len(frame_keys)
    print(f"\nDone. {T} frames encoded.")

    # ------------------------------------------------------------------
    # Gather video metadata (requires a second cap peek for fps)
    # ------------------------------------------------------------------
    if is_video:
        try:
            import cv2
            cap = cv2.VideoCapture(args.video)
            video_fps_value  = cap.get(cv2.CAP_PROP_FPS) or 30.0
            cap.release()
        except Exception:
            video_fps_value = 0.0
        sample_fps_value = args.fps if args.fps is not None else video_fps_value

    # ------------------------------------------------------------------
    # Pack and save
    # ------------------------------------------------------------------
    patch_tokens = torch.cat(all_tokens, dim=0)   # [T, N, D]
    if args.half:
        patch_tokens = patch_tokens.half()

    proj = (
        visual.proj.detach().cpu()
        if hasattr(visual, "proj") and visual.proj is not None
        else None
    )

    payload: dict = {
        "patch_tokens": patch_tokens,
        "proj":         proj,
        "frame_keys":   frame_keys,
        "model_name":   args.model,
        "image_size":   visual.image_size,
        "patch_size":   visual.patch_size,
        "width":        visual.width,
    }
    if is_video:
        payload.update({
            "source_video":  str(Path(args.video).resolve()),
            "video_fps":     video_fps_value,
            "sample_fps":    sample_fps_value,
            "frame_indices": frame_indices,
        })

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out_path)

    size_mb = patch_tokens.nbytes / 1e6
    N, D    = patch_tokens.shape[1], patch_tokens.shape[2]
    print(f"  patch_tokens : {tuple(patch_tokens.shape)}"
          f"  dtype={patch_tokens.dtype}  ({size_mb:.1f} MB)")
    print(f"  proj         : {tuple(proj.shape) if proj is not None else None}")
    print(f"  grid         : {N} patches ({visual.image_size // visual.patch_size}²)"
          f"  width={D}")
    if is_video:
        dur = T / sample_fps_value if sample_fps_value else 0
        print(f"  video        : {video_fps_value:.3f} fps source"
              f"  →  {sample_fps_value:.3f} fps sampled"
              f"  ≈  {dur:.1f} s")
    print(f"Saved → {out_path}")
