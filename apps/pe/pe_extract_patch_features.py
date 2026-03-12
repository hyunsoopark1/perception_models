"""
PE Patch Feature Extractor  —  offline preprocessing step.

Runs the frozen PE visual encoder over a directory of video frames and saves
per-frame patch tokens to a single .pt file.  This is the expensive step;
run it once, then use pe_track_query.py for fast repeated queries with any
number of bounding-box tracks — no image encoder reload required.

Saved file layout
-----------------
    patch_tokens : Tensor[T, N, D]   — frozen PE spatial patch tokens (float32)
    proj         : Tensor[D, E] | None — visual projection matrix (D → CLIP dim)
    frame_paths  : list[str]          — absolute paths, sorted
    model_name   : str
    image_size   : int                — encoder input resolution (e.g. 448)
    patch_size   : int                — patch side in pixels  (e.g. 14)
    width        : int  (D)           — patch token dimension

Usage
-----
    # Extract from a directory of frames:
    python apps/pe/pe_extract_patch_features.py \\
        --frame-dir /path/to/frames \\
        --out       patch_features.pt \\
        --model     PE-Core-G14-448

    # Custom batch size to fit GPU memory:
    python apps/pe/pe_extract_patch_features.py \\
        --frame-dir /path/to/frames \\
        --out       patch_features.pt \\
        --batch-size 4

    # Smoke test without downloading weights:
    python apps/pe/pe_extract_patch_features.py \\
        --frame-dir /path/to/frames \\
        --out       /tmp/smoke.pt \\
        --no-pretrained
"""

import argparse
import sys
from pathlib import Path

import torch
from PIL import Image as PILImage

# Image extensions considered as video frames
_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _parse_args():
    p = argparse.ArgumentParser(
        description="Extract PE patch tokens from video frames and save to disk.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--frame-dir", required=True, metavar="DIR",
                   help="Directory of frame images (loaded in sorted order).")
    p.add_argument("--out", required=True, metavar="PATH",
                   help="Output .pt file.")
    p.add_argument("--model", default="PE-Core-G14-448",
                   choices=["PE-Core-G14-448", "PE-Core-L14-336",
                            "PE-Core-B16-224", "PE-Core-S16-384", "PE-Core-T16-384"],
                   help="PE model variant (default: PE-Core-G14-448).")
    p.add_argument("--batch-size", type=int, default=8, metavar="N",
                   help="Frames per GPU batch (default: 8).")
    p.add_argument("--checkpoint", type=str, default=None, metavar="PATH",
                   help="Local .pt path for the PE encoder (overrides HF download).")
    p.add_argument("--no-pretrained", action="store_true",
                   help="Skip loading pretrained weights (smoke test).")
    p.add_argument("--half", action="store_true",
                   help="Save patch tokens as float16 to halve disk/RAM usage.")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    # Discover frames
    # ------------------------------------------------------------------
    frame_dir  = Path(args.frame_dir)
    frame_paths = sorted(
        p for p in frame_dir.iterdir() if p.suffix.lower() in _IMG_EXTS
    )
    if not frame_paths:
        sys.exit(f"No images found in {frame_dir}  (extensions: {_IMG_EXTS})")
    T = len(frame_paths)
    print(f"Found {T} frames in {frame_dir}")

    # ------------------------------------------------------------------
    # Extract patch tokens in batches
    # ------------------------------------------------------------------
    all_tokens: list[torch.Tensor] = []

    with torch.no_grad():
        buf: list[torch.Tensor] = []

        def _flush():
            if not buf:
                return
            imgs   = torch.stack(buf).to(device)
            tokens = visual.forward_features(imgs, norm=True, strip_cls_token=True)
            all_tokens.append(tokens.cpu())
            buf.clear()

        for i, fp in enumerate(frame_paths):
            buf.append(preprocess(PILImage.open(fp).convert("RGB")))
            if len(buf) == args.batch_size:
                _flush()
            if (i + 1) % 100 == 0 or (i + 1) == T:
                print(f"\r  {i + 1}/{T}", end="", flush=True)

        _flush()

    print(f"\nDone encoding.")

    patch_tokens = torch.cat(all_tokens, dim=0)   # [T, N, D]
    if args.half:
        patch_tokens = patch_tokens.half()

    proj = (
        visual.proj.detach().cpu()
        if hasattr(visual, "proj") and visual.proj is not None
        else None
    )

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "patch_tokens": patch_tokens,          # [T, N, D]
        "proj":         proj,                  # [D, E] | None
        "frame_paths":  [str(p) for p in frame_paths],
        "model_name":   args.model,
        "image_size":   visual.image_size,
        "patch_size":   visual.patch_size,
        "width":        visual.width,
    }
    torch.save(payload, out_path)

    size_mb = patch_tokens.nbytes / 1e6
    N       = patch_tokens.shape[1]
    D       = patch_tokens.shape[2]
    print(f"  patch_tokens : {patch_tokens.shape}  dtype={patch_tokens.dtype}  ({size_mb:.1f} MB)")
    print(f"  proj         : {proj.shape if proj is not None else None}")
    print(f"  grid         : {N} patches ({visual.image_size // visual.patch_size}²),  width={D}")
    print(f"Saved → {out_path}")
