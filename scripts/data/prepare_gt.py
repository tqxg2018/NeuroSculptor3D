"""Prepare ground truth for evaluation from ShapeNetCore v2 (which we cannot redistribute).

For every object of a split this writes <out>/<cat>_<id>/
    gt_{0..5}.jpg  six evaluation views (paper protocol, see below)
    mesh.ply       the mesh normalised like the TRELLIS data toolkit: y-up -> z-up, longest bbox side = 1,
                   bbox centred at the origin (matches the Blender-exported meshes for 100/104 test objects)

Views: pyrender, 512x512, camera at (0, 0, 2.5) looking down -z with a 60 deg field of view, one directional light,
object (ShapeNetCore v2 `model_normalized.obj`, y-up) rotated about the vertical axis through its centroid;
a 300-frame turntable is rendered and frames {0, 60, 120, 179, 239, 299} (~every 60 deg) are kept.

    PYOPENGL_PLATFORM=egl python scripts/data/prepare_gt.py --shapenet_root /path/to/ShapeNetCore.v2 \
        --split data/splits/core_test_list.txt --out data/gt/core_test
"""
import argparse
import os

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np  # noqa: E402
import pyrender  # noqa: E402
import trimesh  # noqa: E402
from PIL import Image  # noqa: E402
from tqdm import tqdm  # noqa: E402

import sys  # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from neurosculptor3d.data.shapenet import normalized_mesh  # noqa: E402


def six_view_indices(num_frames=300):
    return sorted(set([0] + [round(j * (num_frames - 1) / 5) for j in range(1, 6)]))


def render_views(obj_path, bg=0, resolution=512, num_frames=300, keep=None, cam_dist=2.5):
    scene_or_mesh = trimesh.load(obj_path, process=False)
    if isinstance(scene_or_mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate([g for g in scene_or_mesh.geometry.values()])
    else:
        mesh = scene_or_mesh
    scene = pyrender.Scene(bg_color=[bg, bg, bg, 255])
    node = scene.add(pyrender.Mesh.from_trimesh(mesh, smooth=True))
    scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=3.0), pose=np.eye(4))
    cam_pose = np.eye(4)
    cam_pose[2, 3] = cam_dist
    scene.add(pyrender.PerspectiveCamera(yfov=np.pi / 3.0), pose=cam_pose)
    renderer = pyrender.OffscreenRenderer(resolution, resolution)
    keep = keep if keep is not None else six_view_indices(num_frames)
    frames = []
    for i in keep:
        angle = 2 * np.pi * i / num_frames
        scene.set_pose(node, pose=trimesh.transformations.rotation_matrix(angle, [0, 1, 0], point=mesh.centroid))
        color, _ = renderer.render(scene)
        frames.append(color)
    renderer.delete()
    return frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapenet_root", required=True, help="ShapeNetCore.v2 root (<synset>/<id>/models/model_normalized.obj)")
    ap.add_argument("--split", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bg", type=int, default=0, help="background gray level (paper protocol: 0 = black)")
    ap.add_argument("--ext", default="jpg", choices=["jpg", "png"])
    ap.add_argument("--cam_dist", type=float, default=2.5, help="camera distance (paper protocol: 2.5)")
    args = ap.parse_args()

    ids = [l.strip() for l in open(args.split) if l.strip()]
    for obj in tqdm(ids):
        od = os.path.join(args.out, obj.replace("/", "_"))
        if os.path.exists(os.path.join(od, f"gt_5.{args.ext}")) and os.path.exists(os.path.join(od, "mesh.ply")):
            continue
        os.makedirs(od, exist_ok=True)
        obj_path = os.path.join(args.shapenet_root, obj, "models", "model_normalized.obj")
        frames = render_views(obj_path, bg=args.bg, cam_dist=args.cam_dist)
        normalized_mesh(obj_path).export(os.path.join(od, "mesh.ply"))
        for k, fr in enumerate(frames):
            Image.fromarray(fr).save(os.path.join(od, f"gt_{k}.{args.ext}"))


if __name__ == "__main__":
    main()
