"""Evaluate reconstructions: 2-way / 10-way top-1, LPIPS, SSIM (image level) and FPD, CD, EMD (structure level).

    python scripts/evaluate.py --pred results/ss_sc_sub01 --gt data/gt/core_test \
        --split data/splits/core_test_list.txt --fpd_ckpt checkpoints/pointnet_fpd.pth

Expected layout
    <pred>/<cat>_<id>/render_{0..5}.jpg, gaussian.ply   (written by scripts/inference.py)
    <gt>/<cat>_<id>/gt_{0..5}.jpg, mesh.ply             (written by scripts/data/prepare_gt.py)

`--protocol paper` (default) reproduces the evaluation used for the paper tables: LPIPS on [0,1] inputs, point
clouds compared in their native frames without normalisation, 3D-Gaussian centres as the predicted point cloud.
`--protocol corrected` uses LPIPS on [-1,1] inputs, rotates the z-up GT meshes into the y-up frame of the
predictions and normalises both point clouds into the unit sphere.
"""
import argparse
import json
import os
import sys

import numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from neurosculptor3d.metrics import semantic, structural  # noqa: E402

PROTOCOLS = {
    "paper": dict(lpips_input="01", normalize="none", align="none"),
    "corrected": dict(lpips_input="m11", normalize="unit_sphere", align="zup_to_yup"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--fpd_ckpt", default=None, help="PointNet checkpoint for FPD (skipped if not given)")
    ap.add_argument("--protocol", default="paper", choices=list(PROTOCOLS))
    ap.add_argument("--pred_points", default="gaussian.ply", help="file inside each prediction folder to sample")
    ap.add_argument("--num_points", type=int, default=2048)
    ap.add_argument("--trials", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip_semantic", action="store_true")
    ap.add_argument("--skip_structure", action="store_true")
    ap.add_argument("--out_json", default=None)
    args = ap.parse_args()
    proto = PROTOCOLS[args.protocol]

    objs = [l.strip().replace("/", "_") for l in open(args.split) if l.strip()]
    missing = [o for o in objs if not os.path.isdir(os.path.join(args.pred, o))]
    if missing:
        raise FileNotFoundError(f"{len(missing)} predictions missing, e.g. {missing[:3]}")
    results = {"protocol": args.protocol, **proto, "num_objects": len(objs)}

    if not args.skip_semantic:
        def find(folder, stem, k):
            for ext in ("jpg", "png"):
                p = os.path.join(folder, f"{stem}_{k}.{ext}")
                if os.path.exists(p):
                    return p
            raise FileNotFoundError(os.path.join(folder, f"{stem}_{k}.*"))
        preds, gts = [], []
        for o in objs:
            for k in range(6):
                preds.append(np.array(Image.open(find(os.path.join(args.pred, o), "render", k)).convert("RGB")))
                gts.append(np.array(Image.open(find(os.path.join(args.gt, o), "gt", k)).convert("RGB")))
        clf = semantic.NWayTop1()
        gt_ids = clf.probs(gts).argmax(-1).numpy()
        pred_probs = clf.probs(preds)
        del clf
        results["2-way"] = float(np.mean(semantic.NWayTop1.accuracy(pred_probs, gt_ids, 2, args.trials, args.seed)))
        results["10-way"] = float(np.mean(semantic.NWayTop1.accuracy(pred_probs, gt_ids, 10, args.trials, args.seed)))
        results["LPIPS"] = float(np.mean(semantic.LPIPSMetric(input_range=proto["lpips_input"])(preds, gts)))
        results["SSIM"] = float(np.mean(semantic.ssim_scores(preds, gts)))

    if not args.skip_structure:
        np.random.seed(args.seed)
        P = np.stack([structural.sample_points(os.path.join(args.pred, o, args.pred_points), args.num_points)
                      for o in tqdm(objs, desc="sample pred")])
        G = np.stack([structural.sample_points(os.path.join(args.gt, o, "mesh.ply"), args.num_points)
                      for o in tqdm(objs, desc="sample gt")])
        if proto["align"] == "zup_to_yup":
            G = G @ structural.Z_UP_TO_Y_UP.T
        if proto["normalize"] == "unit_sphere":
            P = np.stack([structural.normalize_unit_sphere(p) for p in P])
            G = np.stack([structural.normalize_unit_sphere(g) for g in G])
        results["CD"] = float(np.mean([structural.chamfer(g, p) for p, g in zip(P, G)]))
        results["EMD"] = float(np.mean(structural.emd(P, G)))
        if args.fpd_ckpt:
            results["FPD"] = structural.FPD(args.fpd_ckpt)(P, G)

    # paper tables use the MinD-3D scaling: FPD x 1e-1, CD x 1e2, EMD x 1e2
    scaled = {k: v for k, v in results.items() if k in ("2-way", "10-way", "LPIPS", "SSIM")}
    for k, s in (("FPD", 0.1), ("CD", 100.), ("EMD", 100.)):
        if k in results:
            scaled[k] = results[k] * s
    results["table"] = scaled
    print(json.dumps(results, indent=2))
    out = args.out_json or os.path.join(args.pred, f"metrics_{args.protocol}.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
