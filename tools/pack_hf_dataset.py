"""Maintainer tool: stage the Hugging Face dataset release from a local data folder.

    python tools/pack_hf_dataset.py --data data --frames /path/to/frames --frames_518 /path/to/frames_518 \
        --out /path/to/NeuroSculptor3D-data
"""
import argparse
import os
import shutil
import tarfile

import h5py
import numpy as np

SUBJECTS = ["0001", "0002", "0003", "0004", "0005", "0006", "0007", "0008", "0009", "0011", "0012", "0013"]
DESCRIPTION = ("fMRI-Shape single-trial responses. fMRIPrep (volumetric) -> union of the eight NSD nsdgeneral masks "
               "registered to the participant with ANTs -> GLMsingle single-trial betas. "
               "vox: (num_trials, num_voxels) float32; stimuli: presented video per trial.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--frames", required=True, help="<cat>/<id>/<v>.jpg")
    ap.add_argument("--frames_518", required=True, help="<cat>/<id>/<v>.png (scripts/data/preprocess_frames.py)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(os.path.join(args.out, "fmri"), exist_ok=True)
    os.makedirs(os.path.join(args.out, "splits"), exist_ok=True)

    for s in SUBJECTS:
        src = os.path.join(args.data, "fmri", f"sub-{s}.hdf5")
        dst = os.path.join(args.out, "fmri", f"sub-{s}.hdf5")
        with h5py.File(src, "r") as fi, h5py.File(dst, "w") as fo:
            vox = fi["vox"][:]
            fo.create_dataset("vox", data=vox)
            fo.create_dataset("stimuli", data=fi["stimuli"][:])
            fo.attrs["subject"] = f"sub-{s}"
            fo.attrs["description"] = DESCRIPTION
        with h5py.File(dst, "r") as fo:
            assert np.array_equal(fo["vox"][:], vox)
        print(dst, vox.shape)

    for f in ["core_train_list.txt", "core_test_list.txt", "apt_sub0009_list.txt", "apact_sub0011_list.txt"]:
        shutil.copy(os.path.join(args.data, "splits", f), os.path.join(args.out, "splits", f))
    objs = sorted({l.strip() for f in ["core_train_list.txt", "core_test_list.txt", "apact_sub0011_list.txt"]
                   for l in open(os.path.join(args.data, "splits", f)) if l.strip()})

    with tarfile.open(os.path.join(args.out, "frames.tar"), "w") as tar:
        for o in objs:
            for v in range(32):
                tar.add(os.path.join(args.frames, o, f"{v}.jpg"), arcname=f"frames/{o}/{v}.jpg")
    with tarfile.open(os.path.join(args.out, "frames_518.tar"), "w") as tar:
        for o in objs:
            for v in range(32):
                tar.add(os.path.join(args.frames_518, o, f"{v}.png"), arcname=f"frames_518/{o}/{v}.png")
    with tarfile.open(os.path.join(args.out, "ss_latents.tar"), "w") as tar:
        for o in objs:
            tar.add(os.path.join(args.data, "ss_latents", o, "latent.npz"), arcname=f"ss_latents/{o}/latent.npz")
    print(f"{len(objs)} objects packed")


if __name__ == "__main__":
    main()
