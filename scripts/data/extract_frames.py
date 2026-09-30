"""Extract the M=32 viewpoint frames of every fMRI-Shape stimulus video.

Frames are taken at np.linspace(0, n_frames-1, 32) (the videos have 192 frames: 0, 6, 12, ..., 191),
resized to 224x224 and stored as <out>/<cat>/<id>/<v>.jpg, v = 0..31 (viewpoint id).

    python scripts/data/extract_frames.py --stimuli data/fmri-shape/stimuli --out data/frames
"""
import argparse
import os

import cv2
import numpy as np
from tqdm import tqdm


def extract(video_path, out_dir, num_frames=32, size=224):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total == 0 or not cap.isOpened():
        raise ValueError(f"cannot read {video_path}")
    os.makedirs(out_dir, exist_ok=True)
    for vid, idx in enumerate(np.linspace(0, total - 1, num_frames, dtype=int)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f"failed to read frame {idx} of {video_path}")
        cv2.imwrite(os.path.join(out_dir, f"{vid}.jpg"), cv2.resize(frame, (size, size)))
    cap.release()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stimuli", required=True, help="folder with <cat>/<id>.mp4")
    ap.add_argument("--out", required=True)
    ap.add_argument("--num_frames", type=int, default=32)
    args = ap.parse_args()
    videos = sorted(os.path.join(c, v) for c in os.listdir(args.stimuli) if os.path.isdir(os.path.join(args.stimuli, c))
                    for v in os.listdir(os.path.join(args.stimuli, c)) if v.endswith(".mp4"))
    for rel in tqdm(videos):
        out_dir = os.path.join(args.out, rel[:-4])
        if os.path.exists(os.path.join(out_dir, f"{args.num_frames - 1}.jpg")):
            continue
        extract(os.path.join(args.stimuli, rel), out_dir, args.num_frames)


if __name__ == "__main__":
    main()
