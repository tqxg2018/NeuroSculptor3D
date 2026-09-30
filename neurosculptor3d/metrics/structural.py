"""Point-cloud metrics between reconstructions and ground-truth meshes: FPD, Chamfer distance, EMD.

Protocol knobs (see EVALUATION.md):
  normalize: 'none' (paper numbers) | 'unit_sphere' (center + scale into a sphere of radius 0.5, as in MinD-3D)
  align    : 'none' (paper numbers) | 'zup_to_yup'  (rotate the z-up GT meshes into the y-up frame of the
             TRELLIS exports, removing the 90-degree frame mismatch)
Reported numbers use the MinD-3D scaling: FPD x 1e-1, CD x 1e2, EMD x 1e2.
"""
import os
import sys

import numpy as np
import torch
import trimesh
from scipy.linalg import sqrtm
from scipy.spatial import cKDTree

Z_UP_TO_Y_UP = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float64)   # same as TRELLIS `to_glb`/`save_ply`


def sample_points(path: str, n: int = 2048) -> np.ndarray:
    """Surface samples of a mesh, or a random subset of a point cloud (e.g. 3D-Gaussian centres)."""
    m = trimesh.load(path, process=False)
    if isinstance(m, trimesh.Scene):
        m = trimesh.util.concatenate([g for g in m.geometry.values()])
    if isinstance(m, trimesh.Trimesh) and m.faces is not None and len(m.faces) > 0:
        return trimesh.sample.sample_surface(m, n)[0]
    pts = np.asarray(m.vertices)
    return pts[np.random.choice(len(pts), n, replace=len(pts) < n)]


def normalize_unit_sphere(pc: np.ndarray) -> np.ndarray:
    pc = pc - pc.mean(0)
    return pc / (2 * np.max(np.sqrt((pc ** 2).sum(1))))


def chamfer(a: np.ndarray, b: np.ndarray) -> float:
    """Symmetric Chamfer distance: mean squared NN distance a->b plus b->a."""
    d1, _ = cKDTree(b).query(a)
    d2, _ = cKDTree(a).query(b)
    return float(np.mean(d1 ** 2) + np.mean(d2 ** 2))


def emd(pred: np.ndarray, gt: np.ndarray, device="cuda") -> np.ndarray:
    """Approximate EMD (Fan et al.), CUDA kernel from third_party/PyTorchEMD, divided by the number of points."""
    root = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                        "third_party", "PyTorchEMD")
    if root not in sys.path:
        sys.path.insert(0, root)
    from emd import earth_mover_distance
    p = torch.from_numpy(pred).permute(0, 2, 1).to(device)
    g = torch.from_numpy(gt).permute(0, 2, 1).to(device)
    return (earth_mover_distance(g, p, transpose=True) / pred.shape[1]).cpu().numpy()


class FPD:
    def __init__(self, ckpt: str, device="cuda"):
        from .pointnet import PointNetCls
        self.net = PointNetCls(k=13)
        self.net.load_state_dict(torch.load(ckpt, map_location="cpu")["model_state_dict"])
        self.net = self.net.to(device).eval()
        self.device = device

    @torch.no_grad()
    def activations(self, pcs: np.ndarray, bs: int = 60) -> np.ndarray:
        acts = []
        for i in range(0, len(pcs), bs):
            x = torch.from_numpy(pcs[i:i + bs]).permute(0, 2, 1).float().to(self.device)
            acts.append(self.net(x)[1].cpu().numpy())
        return np.concatenate(acts)

    def __call__(self, pred: np.ndarray, gt: np.ndarray, eps: float = 1e-4) -> float:
        a, b = self.activations(pred), self.activations(gt)
        mu1, s1 = a.mean(0), np.cov(a, rowvar=False)
        mu2, s2 = b.mean(0), np.cov(b, rowvar=False)
        covmean, _ = sqrtm(s1.dot(s2), disp=False)
        if not np.isfinite(covmean).all():
            off = np.eye(s1.shape[0]) * eps
            covmean = sqrtm((s1 + off).dot(s2 + off))
        covmean = covmean.real
        diff = mu1 - mu2
        return float(diff.dot(diff) + np.trace(s1) + np.trace(s2) - 2 * np.trace(covmean))
