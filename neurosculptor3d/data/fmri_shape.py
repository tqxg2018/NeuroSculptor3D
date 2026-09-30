"""fMRI-Shape dataset for NeuroSculptor3D.

Expected layout (see DATA.md):
    <fmri_h5>                      preprocessed single-trial betas, datasets:
                                     'vox'      (num_trials, num_voxels) float32
                                     'stimuli'  (num_trials,) bytes, e.g. b"./stimuli/02691156/<id>.mp4"
    <feature_root>/<cat>/<id>/<v>.npy   DINOv2 tokens of frame v (v = 0..31), shape (1, 1374, 1024)
    <latent_root>/<cat>/<id>/latent.npz TRELLIS sparse-structure latent, key 'mean', shape (8, 16, 16, 16)
    <splits>/core_{train,test}_list.txt  one "<cat>/<id>" per line
"""
import os
import random
from typing import Dict, List, Optional

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from .shapenet import SYNSET_TO_NAME


def read_split(path: str) -> List[str]:
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def stimulus_key(obj_id: str) -> str:
    """'02691156/<id>' -> './stimuli/02691156/<id>.mp4' (naming used inside the fMRI hdf5 files)."""
    return f"./stimuli/{obj_id}.mp4"


def group_voxels(voxels: np.ndarray, num_keyvoxel: int = 512, num_neighbor: int = 32) -> np.ndarray:
    """Training-free voxel grouping of UniBrain (Wang et al., 2024).

    `num_keyvoxel` anchors are spread uniformly over the (spatially ordered) voxel vector and each group takes the
    `num_neighbor` consecutive voxels around its anchor, which maps any number of voxels to a fixed (G, K) array.
    voxels: (..., V) -> (..., G, K)
    """
    length = voxels.shape[-1]
    half = num_neighbor // 2
    key = torch.linspace(half, length - half, num_keyvoxel).long().clamp(half, length - half)
    idx = (key[:, None] + torch.arange(-half, half)[None]).clamp(0, length - 1).numpy()
    return voxels[..., idx]


class FMRIShapeDataset(Dataset):
    """One item per stimulus.

    split='train': one item per training stimulus (single presentation); a random viewpoint is drawn per access.
    split='test' : one item per test stimulus; repeated presentations are averaged (after z-scoring).

    zscore:
        'train' - voxel-wise mean/std of this subject's training trials (paper setting for SS-SC)
        'self'  - mean/std over all trials in the file (for new subjects without training data, NS-SC / NS-NC)
        'none'  - no normalisation
    """

    def __init__(self, fmri_h5: str, split: str, obj_ids: List[str], train_ids: Optional[List[str]] = None,
                 feature_root: Optional[str] = None, latent_root: Optional[str] = None, num_views: int = 32,
                 num_keyvoxel: int = 512, num_neighbor: int = 32, zscore: str = "train",
                 average_repeats: bool = True, fixed_view: Optional[int] = None, seed: int = 0):
        assert split in ("train", "test")
        self.split, self.num_views, self.fixed_view, self.seed = split, num_views, fixed_view, seed
        self.feature_root, self.latent_root = feature_root, latent_root

        with h5py.File(fmri_h5, "r") as h:
            vox = h["vox"][:].astype(np.float32)
            stimuli = [s.decode("utf-8") if isinstance(s, bytes) else str(s) for s in h["stimuli"][:]]
        self.num_voxels = vox.shape[1]

        if zscore == "train":
            assert train_ids, "zscore='train' needs the training split of this subject"
            rows = [stimuli.index(stimulus_key(o)) for o in train_ids]   # first occurrence, as in the paper code
            mean, std = vox[rows].mean(0), vox[rows].std(0)
            vox = (vox - mean) / (std + 1e-6)
        elif zscore == "self":
            vox = (vox - vox.mean(0)) / (vox.std(0) + 1e-6)
        elif zscore != "none":
            raise ValueError(zscore)

        self.obj_ids, voxels = [], []
        for o in obj_ids:
            rows = [i for i, s in enumerate(stimuli) if s == stimulus_key(o)]
            if not rows:
                raise KeyError(f"{o} not found in {fmri_h5}")
            if split == "train":
                voxels.append(vox[rows[0]])
                self.obj_ids.append(o)
            elif average_repeats:
                voxels.append(vox[rows].mean(0))
                self.obj_ids.append(o)
            else:
                for r in rows:
                    voxels.append(vox[r])
                    self.obj_ids.append(o)
        self.voxels = group_voxels(np.stack(voxels), num_keyvoxel, num_neighbor)[:, None].astype(np.float32)  # (N,1,G,K)

        self.latents: Dict[str, np.ndarray] = {}
        if latent_root is not None:
            for o in set(self.obj_ids):
                self.latents[o] = np.load(os.path.join(latent_root, o, "latent.npz"))["mean"].astype(np.float32)

    def __len__(self):
        return len(self.obj_ids)

    def category_name(self, obj_id: str) -> str:
        return SYNSET_TO_NAME[obj_id.split("/")[0]]

    def _view(self, idx: int) -> int:
        if self.fixed_view is not None:
            return self.fixed_view
        if self.split == "train":
            return random.randint(0, self.num_views - 1)   # python RNG is re-seeded per DataLoader worker
        return int(np.random.RandomState(self.seed + idx).randint(self.num_views))   # deterministic for evaluation

    def load_image_target(self, obj_id: str, view: int) -> np.ndarray:
        feat = np.load(os.path.join(self.feature_root, obj_id, f"{view}.npy"))
        return feat.reshape(feat.shape[-2], feat.shape[-1]).astype(np.float32)        # (1374, 1024)

    def __getitem__(self, idx: int):
        o = self.obj_ids[idx]
        view = self._view(idx)
        item = {"obj_id": o, "text": self.category_name(o), "voxels": torch.from_numpy(self.voxels[idx]),
                "view": view}
        if self.feature_root is not None:
            item["image_target"] = torch.from_numpy(self.load_image_target(o, view))
        if self.latents:
            item["structure_target"] = torch.from_numpy(self.latents[o])
        return item
