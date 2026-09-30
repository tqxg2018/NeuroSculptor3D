"""CPU unit tests:  python -m pytest tests -q"""
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from neurosculptor3d import losses as L  # noqa: E402
from neurosculptor3d.config import load_config  # noqa: E402
from neurosculptor3d.data import group_voxels  # noqa: E402
from neurosculptor3d.models import NeuroSculptor3DEncoder, convert_legacy_state_dict  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_group_voxels_contiguous_windows():
    v = np.arange(1000, dtype=np.float32)[None]          # voxel value == voxel index
    g = group_voxels(v, num_keyvoxel=8, num_neighbor=4)
    assert g.shape == (1, 8, 4)
    assert np.all(np.diff(g[0], axis=1) == 1)             # each group is a run of consecutive voxels
    assert g[0, 0, 0] == 0 and g[0, -1, -1] == 999        # anchors span the whole vector


@pytest.mark.parametrize("text,structure", [(True, True), (True, False), (False, True), (False, False)])
def test_encoder_shapes(text, structure):
    torch.manual_seed(0)
    enc = NeuroSculptor3DEncoder(num_tokens=64, hidden_dim=64, image_seq_len=10, image_dim=64, text_seq_len=7,
                                 text_dim=32, structure_guidance=structure, text_guidance=text).eval()
    out = enc(torch.randn(2, 1, 512, 32), torch.tensor([0, 31]))
    assert out["image"].shape == (2, 10, 64) and out["image_proj"].shape == (2, 10, 64)
    assert (out["text"] is None) != text
    assert (out["structure"] is None) != structure
    if text:
        assert out["text"].shape == (2, 7, 32)
    if structure:
        assert out["structure"].shape == (2, 8, 16, 16, 16)


def test_legacy_key_conversion():
    enc = NeuroSculptor3DEncoder(num_tokens=64, hidden_dim=64, image_seq_len=10, image_dim=64, text_seq_len=7,
                                 text_dim=32)
    sd = {("time_embed." + k[len("view_embed."):] if k.startswith("view_embed.") else
           "time_projection." + k[len("view_proj."):] if k.startswith("view_proj.") else k): v
          for k, v in enc.state_dict().items()}
    sd["time_embed.weight"] = torch.cat([sd["time_embed.weight"], torch.zeros(160, 64)])   # 192 legacy rows
    enc.load_state_dict(convert_legacy_state_dict(sd, num_views=32), strict=True)


def test_losses_finite():
    torch.manual_seed(0)
    x = torch.randn(4, 1, 512, 32)
    mixed, perm, betas, select = L.mixco(x.clone())
    p, t = torch.nn.functional.normalize(torch.randn(4, 16), dim=-1), torch.nn.functional.normalize(torch.randn(4, 16), dim=-1)
    assert torch.isfinite(L.mixco_nce(p, t, temp=.006, perm=perm, betas=betas, select=select))
    assert torch.isfinite(L.soft_clip_loss(p, t, temp=.006))
    assert 0 <= L.dice_loss(torch.rand(2, 1, 8, 8, 8), torch.rand(2, 1, 8, 8, 8)).item() <= 1


def test_config_overrides():
    cfg = load_config(os.path.join(ROOT, "configs", "neurosculptor3d.yaml"),
                      ["model.text_guidance=false", "train.max_lr=1e-4", "data.fmri_h5=data/fmri/sub-0002.hdf5"])
    assert cfg.model.text_guidance is False and cfg.train.max_lr == 1e-4
    assert cfg.data.fmri_h5 == "data/fmri/sub-0002.hdf5" and cfg.loss.image_latent == 10.0
