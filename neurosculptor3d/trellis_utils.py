"""Thin wrappers around (unmodified, upstream) TRELLIS.

TRELLIS is not pip-installable, so its repository root must be importable. We look for it in
$TRELLIS_ROOT, then ./third_party/TRELLIS (git submodule).
"""
import os
import sys
from typing import List, Optional

import numpy as np
import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def setup_trellis(trellis_root: Optional[str] = None, attn_backend: str = "xformers", spconv_algo: str = "native"):
    os.environ.setdefault("ATTN_BACKEND", attn_backend)
    os.environ.setdefault("SPCONV_ALGO", spconv_algo)
    root = trellis_root or os.environ.get("TRELLIS_ROOT") or os.path.join(_REPO_ROOT, "third_party", "TRELLIS")
    if not os.path.isdir(os.path.join(root, "trellis")):
        raise FileNotFoundError(f"TRELLIS not found at {root}; run `git submodule update --init --recursive` "
                                f"or set TRELLIS_ROOT")
    if root not in sys.path:
        sys.path.insert(0, root)
    return root


class CLIPTextEncoder(torch.nn.Module):
    """Frozen CLIP ViT-L/14 text encoder, identical to `TrellisTextTo3DPipeline.encode_text` (77 x 768 tokens)."""

    def __init__(self, name: str = "openai/clip-vit-large-patch14"):
        super().__init__()
        from transformers import AutoTokenizer, CLIPTextModel
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.model = CLIPTextModel.from_pretrained(name).eval().requires_grad_(False)

    @torch.no_grad()
    def forward(self, text: List[str]) -> torch.Tensor:
        tokens = self.tokenizer(text, max_length=77, padding="max_length", truncation=True, return_tensors="pt")
        return self.model(input_ids=tokens["input_ids"].to(self.model.device)).last_hidden_state


def load_ss_decoder(trellis_image_ckpt: str):
    """TRELLIS sparse-structure VAE decoder: (B, 8, 16, 16, 16) latent -> (B, 1, 64, 64, 64) occupancy logits."""
    import trellis.models as models
    return models.from_pretrained(os.path.join(trellis_image_ckpt, "ckpts", "ss_dec_conv3d_16l8_fp16")).eval()


def load_ss_encoder(trellis_image_ckpt: str):
    import trellis.models as models
    return models.from_pretrained(os.path.join(trellis_image_ckpt, "ckpts", "ss_enc_conv3d_16l8_fp16")).eval()


def load_image_pipeline(trellis_image_ckpt: str = "JeffreyXiang/TRELLIS-image-large"):
    from trellis.pipelines import TrellisImageTo3DPipeline
    pipe = TrellisImageTo3DPipeline.from_pretrained(trellis_image_ckpt)
    pipe.cuda()
    return pipe


@torch.no_grad()
def encode_frame(pipe, image) -> torch.Tensor:
    """TRELLIS image conditioning for one stimulus frame: rembg + crop + 518px + DINOv2-L/14-reg tokens (1, 1374, 1024)."""
    return pipe.encode_image([pipe.preprocess_image(image)])


@torch.no_grad()
def generate_from_embeddings(pipe, embeddings: torch.Tensor, seed: int = 42, formats=("gaussian", "mesh"),
                             mode: str = "stochastic", sparse_structure_sampler_params: dict = None,
                             slat_sampler_params: dict = None) -> dict:
    """Multi-view TRELLIS generation from (predicted) DINOv2 tokens.

    embeddings: (d, 1374, 1024), one entry per decoded viewpoint. In 'stochastic' mode TRELLIS cycles through the
    d conditions over the sampling steps (upstream `run_multi_image`, but starting from embeddings instead of images).
    """
    sparse_structure_sampler_params = sparse_structure_sampler_params or {}
    slat_sampler_params = slat_sampler_params or {}
    cond = {"cond": embeddings, "neg_cond": torch.zeros_like(embeddings)[:1]}
    torch.manual_seed(seed)
    ss_steps = {**pipe.sparse_structure_sampler_params, **sparse_structure_sampler_params}.get("steps")
    with pipe.inject_sampler_multi_image("sparse_structure_sampler", len(embeddings), ss_steps, mode=mode):
        coords = pipe.sample_sparse_structure(cond, 1, sparse_structure_sampler_params)
    slat_steps = {**pipe.slat_sampler_params, **slat_sampler_params}.get("steps")
    with pipe.inject_sampler_multi_image("slat_sampler", len(embeddings), slat_steps, mode=mode):
        slat = pipe.sample_slat(cond, coords, slat_sampler_params)
    return pipe.decode_slat(slat, list(formats))


def render_turntable(sample, num_frames: int = 300, resolution: int = 512, r: float = 2.0, fov: float = 40.,
                     bg_color=(0, 0, 0), keep: Optional[List[int]] = None) -> List[np.ndarray]:
    """TRELLIS turntable video (yaw 0..2pi, oscillating pitch). `keep` selects frame indices to return."""
    from trellis.utils import render_utils
    frames = render_utils.render_video(sample, resolution=resolution, bg_color=bg_color, num_frames=num_frames,
                                       r=r, fov=fov)["color"]
    return frames if keep is None else [frames[i] for i in keep]


def six_view_indices(num_frames: int = 300) -> List[int]:
    """Frames used for image metrics: one every ~60 degrees -> [0, 60, 120, 179, 239, 299] for 300 frames."""
    return sorted(set([0] + [round(j * (num_frames - 1) / 5) for j in range(1, 6)]))
