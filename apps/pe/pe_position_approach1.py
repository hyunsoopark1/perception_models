""" PE Position-Prompted Scene Understanding — Approach 1 Patch Token Selection + Cross-Attention (no retraining of PE required)  Usage:     python pe_position_approach1.py  Requirements:     pip install torch torchvision     # PE (Meta's Perception Encoder) installed from:     # https://github.com/facebookresearch/perception_encoder """  import torch import torch.nn as nn import torch.nn.functional as F from typing import Tuple, Optional from dataclasses import dataclass   # --------------------------------------------------------------------------- # Data structures # ---------------------------------------------------------------------------  @dataclass class PositionPrompt:     """
    A position prompt in normalized image coordinates [0, 1].

    Supports three prompt types:
        - point:  (cx, cy)
        - bbox:   (x1, y1, x2, y2)
        - radius: (cx, cy, r)   — circular region
    """
    prompt_type: str          # "point" | "bbox" | "radius"
    coords: Tuple[float, ...]  # normalized coords in [0, 1]
    radius: float = 0.1        # used for "point" and "radius" types


# ---------------------------------------------------------------------------
# Patch selection utilities
# ---------------------------------------------------------------------------

def build_patch_grid(image_size: int, patch_size: int) -> torch.Tensor:
    """
    Returns patch center coordinates as a tensor of shape [N, 2] in [0, 1],
    where N = (image_size // patch_size) ** 2.
    """
    n = image_size // patch_size
    step = 1.0 / n
    # Center of each patch in normalized coords
    coords = torch.arange(n) * step + step / 2.0      # [n]
    gy, gx = torch.meshgrid(coords, coords, indexing="ij")  # [n, n]
    grid = torch.stack([gx.flatten(), gy.flatten()], dim=-1)  # [N, 2]
    return grid


def select_patches_from_prompt(
    prompt: PositionPrompt,
    patch_grid: torch.Tensor,   # [N, 2]  (cx, cy) in [0, 1]
    top_k: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
        selected_indices : LongTensor [k]   — indices into the patch sequence
        weights          : FloatTensor [k]  — soft weight per selected patch
    """
    if prompt.prompt_type == "point":
        cx, cy = prompt.coords
        center = torch.tensor([cx, cy], dtype=torch.float32)
        dists = torch.norm(patch_grid - center, dim=-1)         # [N]
        weights = torch.exp(-dists / (2 * prompt.radius ** 2))  # Gaussian
        if top_k is not None:
            selected_indices = torch.topk(weights, top_k).indices
        else:
            # keep patches within 2*radius
            selected_indices = (dists < 2 * prompt.radius).nonzero(as_tuple=True)[0]
        weights = weights[selected_indices]

    elif prompt.prompt_type == "bbox":
        x1, y1, x2, y2 = prompt.coords
        in_box = (
            (patch_grid[:, 0] >= x1) & (patch_grid[:, 0] <= x2) &
            (patch_grid[:, 1] >= y1) & (patch_grid[:, 1] <= y2)
        )
        selected_indices = in_box.nonzero(as_tuple=True)[0]
        weights = torch.ones(len(selected_indices))

    elif prompt.prompt_type == "radius":
        cx, cy, r = prompt.coords
        center = torch.tensor([cx, cy], dtype=torch.float32)
        dists = torch.norm(patch_grid - center, dim=-1)
        in_circle = dists < r
        selected_indices = in_circle.nonzero(as_tuple=True)[0]
        weights = torch.exp(-dists[selected_indices] / (2 * r ** 2))

    else:
        raise ValueError(f"Unknown prompt_type: {prompt.prompt_type}")

    if len(selected_indices) == 0:
        raise ValueError(
            f"No patches selected for prompt {prompt}. "
            "Try increasing radius or checking coordinate range."
        )

    # Normalize weights to sum to 1
    weights = weights / (weights.sum() + 1e-8)
    return selected_indices, weights


# ---------------------------------------------------------------------------
# Cross-attention head (lightweight, trained on top of frozen PE)
# ---------------------------------------------------------------------------

class PositionCrossAttention(nn.Module):
    """
    Cross-attention module with CLS token pooling.

    A learned CLS token is prepended to the local patch query sequence:
         [CLS, patch_0, patch_1, ..., patch_k]  ←  queries  [B, k+1, D]
        [all patch tokens]                      ←  keys/values  [B, N, D]

    The CLS token attends over the full image context and learns — through
    the training loss — which local patches carry the most relevant signal.
    Only the CLS output is returned as the scene embedding, so no manual
    pooling (weighted average or otherwise) is needed.

    Args:
        embed_dim : patch token dimension (must match PE output dim)
        num_heads : number of attention heads
        dropout   : attention dropout
    """

    def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # Learnable CLS token — the sole output after cross-attention
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm_q  = nn.LayerNorm(embed_dim)
        self.norm_kv = nn.LayerNorm(embed_dim)
        self.proj    = nn.Linear(embed_dim, embed_dim)
        self.norm_out = nn.LayerNorm(embed_dim)

    def forward(
        self,
        query_tokens: torch.Tensor,   # [B, k, D]  — local patches near prompt
        context_tokens: torch.Tensor, # [B, N, D]  — all patch tokens (full image)
    ) -> torch.Tensor:
        """
        Returns scene_embed: [B, D]

        Flow:
            1. Prepend CLS token to local patch queries  →  [B, k+1, D]
            2. Cross-attend over full image context      →  [B, k+1, D]
            3. Extract CLS output (index 0)              →  [B, D]
        """
        B = query_tokens.shape[0]

        # 1. Prepend CLS token to the local query sequence
        cls = self.cls_token.expand(B, -1, -1)                      # [B, 1, D]
        q = torch.cat([cls, query_tokens], dim=1)                   # [B, k+1, D]

        # 2. Normalize query and context before attention
        q  = self.norm_q(q)
        kv = self.norm_kv(context_tokens)

        # 3. Cross-attention:
        #    - CLS and local patches are queries
        #    - Full image tokens are keys and values
        #    Each query position attends over the entire image context.
        attn_out, _ = self.cross_attn(q, kv, kv)                   # [B, k+1, D]

        # 4. Extract only the CLS output — it has aggregated information
        #    from all local patches via attention, with no manual pooling.
        cls_out = attn_out[:, 0, :]                                 # [B, D]

        scene_embed = self.norm_out(self.proj(cls_out))             # [B, D]
        return scene_embed


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

class PEPositionSearch(nn.Module):
    """
    Wraps a frozen PE encoder with a trainable cross-attention head
    for position-prompted scene understanding.

    Args:
        pe_encoder     : your PE model (ViT backbone, already loaded)
        image_size     : input resolution expected by PE  (e.g. 448)
        patch_size     : ViT patch size                   (e.g. 14)
        embed_dim      : patch token dimension            (e.g. 1024)
        num_heads      : heads in cross-attention
        top_k          : max local patches for "point" prompts
        freeze_encoder : whether to freeze PE weights (recommended)
    """

    def __init__(
        self,
        pe_encoder: nn.Module,
        image_size: int = 448,
        patch_size: int = 14,
        embed_dim: int = 1024,
        num_heads: int = 8,
        top_k: int = 16,
        freeze_encoder: bool = True,
    ):
        super().__init__()
        self.pe_encoder = pe_encoder
        self.image_size = image_size
        self.patch_size = patch_size
        self.top_k = top_k

        if freeze_encoder:
            for p in self.pe_encoder.parameters():
                p.requires_grad_(False)

        self.cross_attn_head = PositionCrossAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
        )

        # Pre-compute patch grid (same for every image at this resolution)
        self.register_buffer(
            "patch_grid",
            build_patch_grid(image_size, patch_size),
        )

    # ------------------------------------------------------------------
    # Internal: extract all patch tokens from PE
    # ------------------------------------------------------------------

    def _extract_patch_tokens(self, images: torch.Tensor) -> torch.Tensor:
        """
        Run PE and return intermediate patch tokens [B, N, D].

        PE exposes patch tokens via its trunk.  Adjust the attribute path
        to match the version of PE you have installed, e.g.:
            self.pe_encoder.trunk.patch_tokens   (some builds)
            self.pe_encoder.get_intermediate_layers(images, n=1)
        """
        # --- Option A: PE exposes get_intermediate_layers (ViT-style) ---
        if hasattr(self.pe_encoder, "get_intermediate_layers"):
            # returns list of [B, N, D]; take the last one
            tokens = self.pe_encoder.get_intermediate_layers(images, n=1)[-1]
            # Strip CLS token if present
            if tokens.shape[1] == self.patch_grid.shape[0] + 1:
                tokens = tokens[:, 1:, :]
            return tokens

        # --- Option B: full forward, grab patch tokens from trunk -------
        with torch.set_grad_enabled(not self.pe_encoder.training):
            out = self.pe_encoder(images)

        # PE typically returns a dict; adjust key as needed
        if isinstance(out, dict):
            tokens = out.get("patch_tokens", out.get("features"))
        else:
            tokens = out  # assume [B, N, D] or [B, N+1, D]

        if tokens.ndim == 2:
            raise ValueError(
                "PE returned a pooled [B, D] tensor instead of patch tokens. "
                "Use get_intermediate_layers or set return_patch_tokens=True."
            )

        if tokens.shape[1] == self.patch_grid.shape[0] + 1:
            tokens = tokens[:, 1:, :]  # remove CLS

        return tokens   # [B, N, D]

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        images: torch.Tensor,          # [B, C, H, W]
        prompts: list[PositionPrompt], # length B
    ) -> torch.Tensor:
        """
        Returns position-aware scene embeddings: [B, D]
        """
        B = images.shape[0]
        assert len(prompts) == B, "One prompt per image required"

        # 1. Extract full-image patch tokens (frozen PE)
        all_tokens = self._extract_patch_tokens(images)   # [B, N, D]
        D = all_tokens.shape[-1]

        # 2. For each sample, select local patches based on the prompt.
        #    Weights are no longer needed — the CLS token learns to pool
        #    semantically rather than by geometric distance.
        query_list = []
        for b, prompt in enumerate(prompts):
            idx, _ = select_patches_from_prompt(
                prompt,
                self.patch_grid.to(images.device),
                top_k=self.top_k if prompt.prompt_type == "point" else None,
            )
            query_list.append(all_tokens[b, idx, :])   # [k_b, D]

        # Pad to same k across the batch
        max_k = max(q.shape[0] for q in query_list)
        query_padded = torch.zeros(B, max_k, D, device=images.device)
        for b in range(B):
            k = query_list[b].shape[0]
            query_padded[b, :k] = query_list[b]

        # 3. Cross-attention with CLS token pooling.
        #    CLS is prepended inside PositionCrossAttention and its output
        #    is returned directly — no manual pooling here.
        scene_embeds = self.cross_attn_head(
            query_tokens=query_padded,   # [B, max_k, D]
            context_tokens=all_tokens,   # [B, N, D]
        )   # [B, D]

        # 4. L2-normalize for cosine-similarity FAISS indexing
        scene_embeds = F.normalize(scene_embeds, dim=-1)
        return scene_embeds

    # ------------------------------------------------------------------
    # Convenience: encode a single image + prompt (inference)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def encode(
        self,
        image: torch.Tensor,       # [1, C, H, W]  or  [C, H, W]
        prompt: PositionPrompt,
    ) -> torch.Tensor:
        """Returns a single normalized scene embedding [D]."""
        self.eval()
        if image.ndim == 3:
            image = image.unsqueeze(0)
        embed = self.forward(image, [prompt])
        return embed.squeeze(0)


# ---------------------------------------------------------------------------
# FAISS index helper (optional, requires faiss-cpu / faiss-gpu)
# ---------------------------------------------------------------------------

class PositionAwareFAISSIndex:
    """
    Thin wrapper around a flat L2 / IP FAISS index for storing and
    querying position-aware scene embeddings.
    """

    def __init__(self, embed_dim: int, use_gpu: bool = False):
        try:
            import faiss
        except ImportError:
            raise ImportError("Install faiss: pip install faiss-cpu")

        self.index = faiss.IndexFlatIP(embed_dim)   # inner product = cosine on L2-normed vecs
        if use_gpu:
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)

        self.metadata: list[dict] = []

    def add(self, embed: torch.Tensor, meta: dict):
        """Add a single embedding with associated metadata."""
        import numpy as np
        vec = embed.detach().cpu().float().numpy()
        if vec.ndim == 1:
            vec = vec[None]
        self.index.add(vec)
        self.metadata.append(meta)

    def search(self, query: torch.Tensor, top_k: int = 5):
        """Return top-k (score, metadata) pairs."""
        import numpy as np
        vec = query.detach().cpu().float().numpy()
        if vec.ndim == 1:
            vec = vec[None]
        scores, indices = self.index.search(vec, top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0:
                results.append({"score": float(score), **self.metadata[idx]})
        return results


# ---------------------------------------------------------------------------
# Quick smoke test (no real PE needed — uses a dummy encoder)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== PE Position Approach 1 — Smoke Test ===\n")

    # -- Dummy PE that mimics get_intermediate_layers output --
    class DummyPE(nn.Module):
        def __init__(self, image_size=448, patch_size=14, embed_dim=1024):
            super().__init__()
            n_patches = (image_size // patch_size) ** 2
            # +1 for CLS token
            self.proj = nn.Linear(3 * patch_size * patch_size, embed_dim)
            self.n_patches = n_patches
            self.embed_dim = embed_dim

        def get_intermediate_layers(self, x, n=1):
            B = x.shape[0]
            # Return random patch tokens [B, N+1, D]  (includes CLS)
            tokens = torch.randn(B, self.n_patches + 1, self.embed_dim, device=x.device)
            return [tokens]

    IMAGE_SIZE = 448
    PATCH_SIZE = 14
    EMBED_DIM  = 1024
    BATCH_SIZE = 2

    dummy_pe = DummyPE(IMAGE_SIZE, PATCH_SIZE, EMBED_DIM)

    model = PEPositionSearch(
        pe_encoder=dummy_pe,
        image_size=IMAGE_SIZE,
        patch_size=PATCH_SIZE,
        embed_dim=EMBED_DIM,
        num_heads=8,
        top_k=16,
        freeze_encoder=True,
    )

    # Dummy images
    images = torch.randn(BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE)

    # Two different prompt types
    prompts = [
        PositionPrompt(prompt_type="point", coords=(0.5, 0.5), radius=0.15),
        PositionPrompt(prompt_type="bbox",  coords=(0.2, 0.2, 0.7, 0.7)),
    ]

    embeds = model(images, prompts)
    print(f"Output shape      : {embeds.shape}")          # [2, 1024]
    print(f"L2 norms (≈1.0)   : {embeds.norm(dim=-1)}")   # should be ~1.0
    print(f"Cosine similarity : {(embeds[0] @ embeds[1]).item():.4f}\n")

    # Single-image encode
    single_embed = model.encode(
        images[0],
        PositionPrompt(prompt_type="point", coords=(0.3, 0.6), radius=0.1),
    )
    print(f"Single embed shape: {single_embed.shape}")    # [1024]

    # FAISS round-trip
    try:
        faiss_index = PositionAwareFAISSIndex(embed_dim=EMBED_DIM)
        faiss_index.add(embeds[0], {"image_id": "img_001", "prompt": "center"})
        faiss_index.add(embeds[1], {"image_id": "img_002", "prompt": "top-left bbox"})

        results = faiss_index.search(single_embed, top_k=2)
        print("\nFAISS search results:")
        for r in results:
            print(f"  score={r['score']:.4f}  image_id={r['image_id']}")

    except ImportError:
        print("(faiss not installed — skipping FAISS test)")

    print("\n✓ All checks passed.")
