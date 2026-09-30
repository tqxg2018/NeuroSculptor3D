"""TRELLIS image preprocessing of the stimulus frames (CPU only): rembg/u2net background removal, crop around the
object (1.2x its bounding box), resize to 518x518, alpha-premultiplied RGB. Saved losslessly as PNG, so that the
DINOv2 targets can be recomputed exactly (scripts/data/extract_dino_features.py --preprocessed).

Identical to `TrellisImageTo3DPipeline.preprocess_image` (TRELLIS, MIT License) for RGB inputs.

    python scripts/data/preprocess_frames.py --frames data/frames --out data/frames_518 --workers 12
"""
import argparse
import os
from multiprocessing import Pool

import numpy as np
from PIL import Image
from tqdm import tqdm

_session = None


def preprocess(image: Image.Image) -> Image.Image:
    global _session
    import rembg
    image = image.convert("RGB")
    scale = min(1, 1024 / max(image.size))
    if scale < 1:
        image = image.resize((int(image.width * scale), int(image.height * scale)), Image.Resampling.LANCZOS)
    if _session is None:
        _session = rembg.new_session("u2net")
    output = rembg.remove(image, session=_session)
    alpha = np.array(output)[:, :, 3]
    bbox = np.argwhere(alpha > 0.8 * 255)
    bbox = np.min(bbox[:, 1]), np.min(bbox[:, 0]), np.max(bbox[:, 1]), np.max(bbox[:, 0])
    center = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    size = int(max(bbox[2] - bbox[0], bbox[3] - bbox[1]) * 1.2)
    bbox = center[0] - size // 2, center[1] - size // 2, center[0] + size // 2, center[1] + size // 2
    output = output.crop(bbox).resize((518, 518), Image.Resampling.LANCZOS)
    output = np.array(output).astype(np.float32) / 255
    output = output[:, :, :3] * output[:, :, 3:4]
    return Image.fromarray((output * 255).astype(np.uint8))


def work(job):
    src, dst = job
    if not os.path.exists(dst):
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        preprocess(Image.open(src)).save(dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True, help="<cat>/<id>/<v>.jpg")
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", nargs="*", default=None)
    ap.add_argument("--num_frames", type=int, default=32)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    if args.split:
        objs = sorted({l.strip() for s in args.split for l in open(s) if l.strip()})
    else:
        objs = sorted(os.path.join(c, o) for c in os.listdir(args.frames) for o in os.listdir(os.path.join(args.frames, c)))
    jobs = [(os.path.join(args.frames, o, f"{v}.jpg"), os.path.join(args.out, o, f"{v}.png"))
            for o in objs for v in range(args.num_frames)]
    with Pool(args.workers) as pool:
        for _ in tqdm(pool.imap_unordered(work, jobs, chunksize=16), total=len(jobs)):
            pass


if __name__ == "__main__":
    main()
