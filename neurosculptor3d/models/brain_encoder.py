"""NeuroSculptor3D brain encoder.

fMRI (grouped voxels) -> global brain embedding -> three decoding paths:
  * perceptual path  : viewpoint-aware DINOv2 patch tokens (1374 x 1024), refined by the diffusion prior
  * semantic path    : CLIP text tokens of the object category (77 x 768)
  * geometric path   : TRELLIS sparse-structure latent (8 x 16 x 16 x 16)

This is a cleaned-up version of `ShapeMind_full_model` from the original research code; the forward
computation is kept numerically identical so that the released checkpoints reproduce the paper setting.
"""
from typing import Dict, Optional

import torch
import torch.nn as nn

from .perceiver import PerceiverResampler


class PerceiverHead(nn.Module):
    """LayerNorm -> Perceiver resampler (learnable queries) -> Linear. Called `CrossAttention_encoder` originally."""

    def __init__(self, patch_embed_dim: int, hidden_size: int, num_latents: int, depth: int = 2):
        super().__init__()
        self.ln = nn.LayerNorm(patch_embed_dim)
        self.proj = nn.Linear(patch_embed_dim, hidden_size)
        self.perceiver = PerceiverResampler(dim=patch_embed_dim, dim_head=96, depth=depth, heads=16,
                                            num_latents=num_latents, num_media_embeds=1)

    def forward(self, x):
        return self.proj(self.perceiver(self.ln(x)))


def _projector(in_dim: int, out_dim: int, h: int) -> nn.Sequential:
    """MLP head producing the embedding used by the contrastive (BiMixCo / SoftCLIP) losses."""
    return nn.Sequential(
        nn.LayerNorm(in_dim), nn.GELU(), nn.Linear(in_dim, h),
        nn.LayerNorm(h), nn.GELU(), nn.Linear(h, h),
        nn.LayerNorm(h), nn.GELU(), nn.Linear(h, out_dim),
    )


def _mlp(in_dim: int, out_dim: int, drop: float) -> nn.Sequential:
    return nn.Sequential(nn.Linear(in_dim, out_dim), nn.GELU(), nn.Dropout(drop), nn.Linear(out_dim, out_dim))


def _conv_block(ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv3d(ch, ch, kernel_size=3, padding=1), nn.BatchNorm3d(ch), nn.GELU(),
        nn.Conv3d(ch, ch, kernel_size=3, padding=1), nn.BatchNorm3d(ch), nn.GELU(),
    )


class NeuroSculptor3DEncoder(nn.Module):
    def __init__(
        self,
        num_keyvoxel: int = 512,       # G: number of voxel groups
        num_neighbor: int = 32,        # K: voxels per group
        group_dim: int = 8,            # per-group embedding size
        hidden_dim: int = 1024,        # D: brain embedding width
        num_tokens: int = 2048,        # H: number of brain tokens
        n_blocks: int = 1,             # number of MLP-mixer blocks
        drop: float = 0.15,
        num_views: int = 32,           # M: number of viewpoints (frames) per stimulus video
        image_seq_len: int = 1374,     # DINOv2-L/14-reg tokens at 518 px (1 cls + 4 reg + 37*37 patches)
        image_dim: int = 1024,
        text_guidance: bool = True,
        text_seq_len: int = 77,
        text_dim: int = 768,
        structure_guidance: bool = True,
        structure_channels: int = 8,
        structure_res: int = 16,
    ):
        super().__init__()
        self.num_keyvoxel, self.num_neighbor = num_keyvoxel, num_neighbor
        self.text_guidance, self.structure_guidance = text_guidance, structure_guidance
        self.structure_channels, self.structure_res = structure_channels, structure_res

        # ---- brain embedder: group mapping + global mapping + token mapping + MLP mixer ----
        self.group_global_mapping = nn.Linear(num_neighbor, group_dim)
        self.global_mapping = nn.Linear(num_keyvoxel * group_dim, hidden_dim)
        self.global_token_mapping = nn.Sequential(nn.Linear(1, num_tokens), nn.LayerNorm(num_tokens), nn.Dropout(0.5))
        self.mixer_blocks1 = nn.ModuleList([nn.Sequential(nn.LayerNorm(hidden_dim), _mlp(hidden_dim, hidden_dim, drop))
                                            for _ in range(n_blocks)])   # mixes the feature dimension
        self.mixer_blocks2 = nn.ModuleList([nn.Sequential(nn.LayerNorm(num_tokens), _mlp(num_tokens, num_tokens, drop))
                                            for _ in range(n_blocks)])   # mixes the token dimension

        # ---- perceptual path (viewpoint-aware) ----
        self.view_embed = nn.Embedding(num_views, image_dim)
        self.view_proj = nn.Linear(num_tokens + 1, num_tokens)
        self.image_initial_encoder = PerceiverHead(hidden_dim, hidden_dim, num_latents=image_seq_len, depth=2)
        self.image_backbone_linear = nn.Linear(hidden_dim, image_dim, bias=True)
        self.image_clip_proj = _projector(image_dim, image_dim, h=image_dim)

        # ---- semantic path ----
        if text_guidance:
            self.text_linear = nn.Linear(hidden_dim, text_dim)
            self.text_initial_encoder = PerceiverHead(text_dim, text_dim, num_latents=text_seq_len, depth=2)
            self.text_backbone_linear = nn.Linear(text_dim, text_dim, bias=True)
            self.text_clip_proj = _projector(text_dim, text_dim, h=text_dim)

        # ---- geometric path ----
        if structure_guidance:
            self.structure_linear_1 = nn.Linear(hidden_dim, structure_channels)
            self.structure_linear_2 = nn.Linear(num_tokens, structure_res ** 3)
            self.structure_backbone = _conv_block(structure_channels)
            self.structure_project = _conv_block(structure_channels)

    def embed_brain(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 1, G, K) grouped voxels -> global brain embedding (B, H, D)."""
        x = self.group_global_mapping(x)
        x = self.global_mapping(x.reshape(x.size(0), -1))
        x = self.global_token_mapping(x.unsqueeze(-1)).transpose(2, 1)
        # MLP-mixer as in MindEye2: note that the token-mixing residual is taken from the block input
        # *before* the feature-mixing step (kept as-is for faithfulness to the paper model)
        residual1 = x
        residual2 = x.permute(0, 2, 1)
        for block1, block2 in zip(self.mixer_blocks1, self.mixer_blocks2):
            x = block1(x) + residual1
            residual1 = x
            x = x.permute(0, 2, 1)
            x = block2(x) + residual2
            residual2 = x
            x = x.permute(0, 2, 1)
        return x

    def perceptual(self, brain: torch.Tensor, view_index: torch.Tensor):
        """Viewpoint-aware brain embedding -> predicted DINOv2 tokens (backbone) and contrastive embedding."""
        v = self.view_embed(view_index).unsqueeze(1)                  # (B, 1, D)
        x = torch.cat((brain, v), dim=1)                              # (B, H + 1, D)
        x = self.view_proj(x.permute(0, 2, 1)).permute(0, 2, 1)      # (B, H, D)
        x = self.image_initial_encoder(x)
        x = self.image_backbone_linear(x)
        return x, self.image_clip_proj(x)

    def semantic(self, brain: torch.Tensor):
        x = self.text_linear(brain)
        x = self.text_initial_encoder(x)
        x = self.text_backbone_linear(x)
        return x, self.text_clip_proj(x)

    def geometric(self, brain: torch.Tensor):
        b = brain.size(0)
        x = self.structure_linear_1(brain)                            # (B, H, C)
        x = self.structure_linear_2(x.permute(0, 2, 1))               # (B, C, R^3)
        # NOTE: the original code reshapes the (B, R^3, C)-ordered tensor into (B, C, R, R, R); this is a fixed
        # (learnable-around) permutation of the entries and is kept unchanged for checkpoint compatibility.
        x = x.permute(0, 2, 1).reshape(b, -1, self.structure_res, self.structure_res, self.structure_res)
        x = self.structure_backbone(x)
        return x, self.structure_project(x)

    def forward(self, x: torch.Tensor, view_index: torch.Tensor) -> Dict[str, Optional[torch.Tensor]]:
        brain = self.embed_brain(x)
        out = {"image": None, "image_proj": None, "text": None, "text_proj": None, "structure": None, "structure_proj": None}
        out["image"], out["image_proj"] = self.perceptual(brain, view_index)
        if self.text_guidance:
            out["text"], out["text_proj"] = self.semantic(brain)
        if self.structure_guidance:
            out["structure"], out["structure_proj"] = self.geometric(brain)
        return out


# key renames between the original research code (`ShapeMind_full_model`) and this module
LEGACY_KEY_MAP = {"time_embed.": "view_embed.", "time_projection.": "view_proj."}


def convert_legacy_state_dict(sd: Dict[str, torch.Tensor], num_views: Optional[int] = None) -> Dict[str, torch.Tensor]:
    """Convert a `backbone.*` state dict saved by the original training script to this module's naming."""
    out = {}
    for k, v in sd.items():
        for old, new in LEGACY_KEY_MAP.items():
            if k.startswith(old):
                k = new + k[len(old):]
        out[k] = v
    if num_views is not None and out["view_embed.weight"].shape[0] > num_views:
        # the paper checkpoint allocated 192 view embeddings but only indices 0..31 are ever used
        out["view_embed.weight"] = out["view_embed.weight"][:num_views].clone()
    return out
