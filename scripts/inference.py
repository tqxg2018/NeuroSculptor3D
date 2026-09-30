"""Reconstruct textured 3D objects from fMRI with a trained NeuroSculptor3D model.

    python scripts/inference.py --ckpt outputs/ss_sc_sub01/model_epoch200.pth --out results/ss_sc_sub01 \
        [--fmri_h5 ... --split ... --zscore self]   # e.g. new subjects (NS-SC / NS-NC)

For every test stimulus: decode d viewpoint-specific DINOv2 token sets (evenly spaced over the 32 frame ids) with
the diffusion prior, feed them to TRELLIS as multi-view conditions and save
    <out>/<cat>_<id>/gaussian.ply        3D Gaussians (y-up, TRELLIS `save_ply` convention)
    <out>/<cat>_<id>/render_{0..5}.jpg  six turntable views (~every 60 deg) used by the image metrics
    <out>/<cat>_<id>/mesh.glb           textured mesh (only with --save_glb, slow)
"""
import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from neurosculptor3d.config import load_config  # noqa: E402
from neurosculptor3d.data import FMRIShapeDataset, read_split  # noqa: E402
from neurosculptor3d.models import NeuroSculptor3DEncoder, build_prior, convert_legacy_state_dict  # noqa: E402
from neurosculptor3d.trellis_utils import (generate_from_embeddings, load_image_pipeline, render_turntable,  # noqa: E402
                                           setup_trellis, six_view_indices)


def evenly_spaced(num: int, length: int = 32):
    return [0] if num == 1 else [round(i * (length - 1) / (num - 1)) for i in range(num)]


def load_model(ckpt_path, cfg, device, legacy=False):
    m, p = cfg.model, cfg.prior
    enc = NeuroSculptor3DEncoder(num_keyvoxel=m.num_keyvoxel, num_neighbor=m.num_neighbor, hidden_dim=m.hidden_dim,
                                 num_tokens=m.num_tokens, n_blocks=m.n_blocks, drop=m.drop, num_views=m.num_views,
                                 text_guidance=m.text_guidance, structure_guidance=m.structure_guidance)
    prior = build_prior(depth=p.depth, dim_head=p.dim_head, timesteps=p.timesteps, cond_drop_prob=p.cond_drop_prob)
    if ckpt_path.endswith(".safetensors"):
        from safetensors.torch import load_file
        sd = load_file(ckpt_path)
    else:
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)["model_state_dict"]
    bb = {k[len("backbone."):]: v for k, v in sd.items() if k.startswith("backbone.")}
    if legacy:
        bb = convert_legacy_state_dict(bb, num_views=m.num_views)
    enc.load_state_dict(bb, strict=True)
    prior.load_state_dict({k[len("diffusion_prior."):]: v for k, v in sd.items() if k.startswith("diffusion_prior.")},
                          strict=True)
    return enc.to(device).eval(), prior.to(device).eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="training checkpoint (.pth) or released model.safetensors")
    ap.add_argument("--config", default=None, help="defaults to the config stored in the checkpoint")
    ap.add_argument("--legacy", action="store_true", help="checkpoint saved by the original research code")
    ap.add_argument("--out", required=True)
    ap.add_argument("--fmri_h5", default=None, help="override data.fmri_h5 (e.g. a new subject)")
    ap.add_argument("--split", default=None, help="override data.test_split")
    ap.add_argument("--zscore", default=None, choices=["train", "self", "none"])
    ap.add_argument("--num_viewpoints", type=int, default=5)
    ap.add_argument("--prior_steps", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--use_gt_features", action="store_true",
                    help="condition TRELLIS on the ground-truth DINOv2 tokens of the same frames (upper bound)")
    ap.add_argument("--bg", type=int, default=0, help="render background gray level (paper: 0 = black)")
    ap.add_argument("--ext", default="jpg", choices=["jpg", "png"], help="render format (paper protocol: jpg)")
    ap.add_argument("--save_glb", action="store_true")
    ap.add_argument("--trellis_root", default=None)
    ap.add_argument("--trellis_ckpt", default=None)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    device = "cuda"
    from neurosculptor3d.config import _to_ns
    if args.config:
        cfg = load_config(args.config)
    elif args.ckpt.endswith(".safetensors"):     # released weights: config.json next to model.safetensors
        cfg = _to_ns(json.load(open(os.path.join(os.path.dirname(args.ckpt), "config.json"))))
    else:
        cfg = _to_ns(torch.load(args.ckpt, map_location="cpu", weights_only=False)["config"])
    d = cfg.data
    fmri_h5 = args.fmri_h5 or d.fmri_h5
    test_ids = read_split(args.split or d.test_split)
    zscore = args.zscore or d.zscore
    dataset = FMRIShapeDataset(fmri_h5, "test", test_ids, train_ids=read_split(d.train_split) if zscore == "train" else None,
                               feature_root=d.feature_root if args.use_gt_features else None, latent_root=None,
                               num_views=d.num_views, num_keyvoxel=cfg.model.num_keyvoxel,
                               num_neighbor=cfg.model.num_neighbor, zscore=zscore)

    setup_trellis(args.trellis_root)
    pipe = load_image_pipeline(args.trellis_ckpt or cfg.trellis.image_ckpt)
    enc, prior = (None, None) if args.use_gt_features else load_model(args.ckpt, cfg, device, legacy=args.legacy)
    views = evenly_spaced(args.num_viewpoints, d.num_views)
    keep = six_view_indices(300)
    print(f"{len(dataset)} stimuli | viewpoints {views} | out {args.out}")

    n = min(len(dataset), args.limit) if args.limit else len(dataset)
    timings = {}
    for i in tqdm(range(n)):
        obj = dataset.obj_ids[i]
        od = os.path.join(args.out, obj.replace("/", "_"))
        if os.path.exists(os.path.join(od, f"render_5.{args.ext}")):
            continue
        os.makedirs(od, exist_ok=True)
        # per-stimulus seed -> results do not depend on processing order
        s = args.seed * 1000 + i
        torch.manual_seed(s); np.random.seed(s % 2 ** 32); random.seed(s)
        torch.cuda.synchronize(); t0 = time.time()
        with torch.no_grad():
            if args.use_gt_features:
                emb = torch.stack([torch.from_numpy(dataset.load_image_target(obj, v)) for v in views]).to(device)
            else:
                x = dataset[i]["voxels"][None].to(device)
                emb = []
                for v in views:
                    tokens = enc(x, torch.tensor([v], device=device))["image"]
                    emb.append(prior.p_sample_loop(tokens.shape, text_cond=dict(text_embed=tokens), cond_scale=1.,
                                                   timesteps=args.prior_steps))
                emb = torch.cat(emb, 0)
            out = generate_from_embeddings(pipe, emb, seed=args.seed,
                                           formats=("gaussian", "mesh") if args.save_glb else ("gaussian",))
        torch.cuda.synchronize(); timings[obj] = time.time() - t0   # fMRI -> 3D (prior decoding + TRELLIS)
        g = out["gaussian"][0]
        g.save_ply(os.path.join(od, "gaussian.ply"))
        for k, fr in enumerate(render_turntable(g, bg_color=(args.bg,) * 3, keep=keep)):
            Image.fromarray(fr).save(os.path.join(od, f"render_{k}.{args.ext}"))
        if args.save_glb:
            from trellis.utils import postprocessing_utils
            with torch.enable_grad():
                glb = postprocessing_utils.to_glb(g, out["mesh"][0], simplify=0.95, texture_size=1024, verbose=False)
            glb.export(os.path.join(od, "mesh.glb"))
    if timings:
        with open(os.path.join(args.out, f"timing_{len(timings)}objs.json"), "w") as f:
            json.dump({"mean_sec": float(np.mean(list(timings.values()))), "per_object": timings}, f, indent=1)


if __name__ == "__main__":
    main()
