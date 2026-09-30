"""Encode ShapeNet meshes into TRELLIS sparse-structure latents (geometric-path targets).

mesh -> AABB-normalised into [-0.5, 0.5]^3 -> 64^3 occupancy (open3d) -> TRELLIS SS-VAE encoder (posterior mean)
-> <out>/<cat>/<id>/latent.npz with key 'mean' of shape (8, 16, 16, 16).

Input meshes are the z-up normalised meshes written by scripts/data/prepare_gt.py (or any mesh in the TRELLIS
convention); use --shapenet_root to normalise the ShapeNet OBJ files on the fly instead.

    python scripts/data/encode_ss_latents.py --shapenet_root /path/to/ShapeNetCore.v2 \
        --split data/splits/core_train_list.txt data/splits/core_test_list.txt --out data/ss_latents
"""
import argparse
import os
import sys

import numpy as np
import open3d as o3d
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from neurosculptor3d.data.shapenet import normalized_mesh  # noqa: E402
from neurosculptor3d.trellis_utils import load_ss_encoder, setup_trellis  # noqa: E402


def voxelize(vertices: np.ndarray, faces: np.ndarray, resolution: int = 64) -> torch.Tensor:
    center = (vertices.min(0) + vertices.max(0)) / 2
    scale = (vertices.max(0) - vertices.min(0)).max()
    v = np.clip((vertices - center) / scale, -0.5 + 1e-6, 0.5 - 1e-6)
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(v), o3d.utility.Vector3iVector(faces))
    grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh_within_bounds(
        mesh, voxel_size=1 / resolution, min_bound=(-0.5, -0.5, -0.5), max_bound=(0.5, 0.5, 0.5))
    idx = np.array([vx.grid_index for vx in grid.get_voxels()])
    ss = torch.zeros(1, 1, resolution, resolution, resolution)
    ss[0, 0, idx[:, 0], idx[:, 1], idx[:, 2]] = 1
    return ss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--mesh_root", help="<mesh_root>/<cat>_<id>/mesh.ply (from prepare_gt.py)")
    src.add_argument("--shapenet_root", help="ShapeNetCore.v2 root, meshes are normalised on the fly")
    ap.add_argument("--trellis_root", default=None)
    ap.add_argument("--trellis_ckpt", default="JeffreyXiang/TRELLIS-image-large")
    args = ap.parse_args()

    setup_trellis(args.trellis_root)
    encoder = load_ss_encoder(args.trellis_ckpt).cuda()
    objs = sorted({l.strip() for s in args.split for l in open(s) if l.strip()})
    for obj in tqdm(objs):
        out = os.path.join(args.out, obj, "latent.npz")
        if os.path.exists(out):
            continue
        if args.mesh_root:
            m = o3d.io.read_triangle_mesh(os.path.join(args.mesh_root, obj.replace("/", "_"), "mesh.ply"))
            vertices, faces = np.asarray(m.vertices), np.asarray(m.triangles)
        else:
            m = normalized_mesh(os.path.join(args.shapenet_root, obj, "models", "model_normalized.obj"))
            vertices, faces = np.asarray(m.vertices), np.asarray(m.faces)
        with torch.no_grad():
            latent = encoder(voxelize(vertices, faces).cuda(), sample_posterior=False)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        np.savez_compressed(out, mean=latent[0].cpu().numpy())


if __name__ == "__main__":
    main()
