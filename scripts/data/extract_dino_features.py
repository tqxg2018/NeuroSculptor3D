"""Encode every stimulus frame into the TRELLIS image-conditioning space (perceptual-path targets).

Each frame goes through TRELLIS `preprocess_image` (rembg/u2net background removal, crop, 518x518) and
`encode_image` (DINOv2 ViT-L/14-reg, layer-normed patch tokens) -> (1, 1374, 1024) float32, saved as
<out>/<cat>/<id>/<v>.npy. ~5.6 MB per frame, ~250 GB for all 1,624 objects x 32 frames; use --fp16 to halve it.

    python scripts/data/extract_dino_features.py --frames data/frames_518 --preprocessed --out data/dino_features

Use the released preprocessed frames (`frames_518`, lossless PNG) to reproduce our targets exactly: background
removal (rembg/onnxruntime) and JPEG decoding (libjpeg vs libjpeg-turbo in different Pillow builds) are not
bit-reproducible across environments. `--frames data/frames` (JPEG) runs the full TRELLIS preprocessing instead.
"""
import argparse
import os
import sys

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from neurosculptor3d.trellis_utils import load_image_pipeline, setup_trellis  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True, help="folder with <cat>/<id>/<v>.jpg (or .png with --preprocessed)")
    ap.add_argument("--preprocessed", action="store_true", help="frames are already TRELLIS-preprocessed 518px PNGs")
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", nargs="*", default=None, help="optional split files restricting the objects")
    ap.add_argument("--num_frames", type=int, default=32)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--save_preprocessed", default=None, help="also store the 518x518 preprocessed frames here")
    ap.add_argument("--trellis_root", default=None)
    ap.add_argument("--trellis_ckpt", default="JeffreyXiang/TRELLIS-image-large")
    args = ap.parse_args()

    if args.split:
        objs = sorted({l.strip() for s in args.split for l in open(s) if l.strip()})
    else:
        objs = sorted(os.path.join(c, o) for c in os.listdir(args.frames) for o in os.listdir(os.path.join(args.frames, c)))
    setup_trellis(args.trellis_root)
    pipe = load_image_pipeline(args.trellis_ckpt)

    for obj in tqdm(objs):
        out_dir = os.path.join(args.out, obj)
        if os.path.exists(os.path.join(out_dir, f"{args.num_frames - 1}.npy")):
            continue
        os.makedirs(out_dir, exist_ok=True)
        for v in range(args.num_frames):
            if args.preprocessed:
                img = Image.open(os.path.join(args.frames, obj, f"{v}.png")).convert("RGB")
            else:
                img = pipe.preprocess_image(Image.open(os.path.join(args.frames, obj, f"{v}.jpg")))
            if args.save_preprocessed:
                os.makedirs(os.path.join(args.save_preprocessed, obj), exist_ok=True)
                img.save(os.path.join(args.save_preprocessed, obj, f"{v}.png"))
            with torch.no_grad():
                feat = pipe.encode_image([img]).cpu().numpy()
            np.save(os.path.join(out_dir, f"{v}.npy"), feat.astype(np.float16) if args.fp16 else feat)


if __name__ == "__main__":
    main()
