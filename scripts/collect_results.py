"""Collect results/<run>/metrics_<protocol>.json into markdown tables (written to results/RESULTS.md)."""
import argparse
import glob
import json
import os

import numpy as np

COLS = ["2-way", "10-way", "LPIPS", "SSIM", "FPD", "CD", "EMD"]


def load(results, run, protocol):
    p = os.path.join(results, run, f"metrics_{protocol}.json")
    return json.load(open(p))["table"] if os.path.exists(p) else None


def runtime(results, run):
    files = glob.glob(os.path.join(results, run, "timing_*objs.json"))
    return json.load(open(files[0]))["mean_sec"] if files else None


def row(name, m, cols=COLS, extra=None):
    cells = ["–" if m is None or c not in m else (f"{m[c]:.3f}") for c in cols]
    if extra is not None:
        cells.append("–" if extra is None else f"{extra:.0f}")
    return f"| {name} | " + " | ".join(cells) + " |"


def header(cols, extra=None):
    h = cols + ([extra] if extra else [])
    return "| | " + " | ".join(h) + " |\n|---|" + "---|" * len(h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    args = ap.parse_args()
    out = []
    for protocol in ("paper", "corrected"):
        out.append(f"## Protocol: {protocol}\n")

        out.append("### Table 1: SS-SC (d = 5)\n")
        out.append(header(COLS))
        subj = []
        for s in range(1, 9):
            m = load(args.results, f"ss_sc_sub-000{s}", protocol)
            out.append(row(f"subject {s}", m))
            if m:
                subj.append(m)
        if subj:
            mean = {c: float(np.mean([m[c] for m in subj if c in m])) for c in COLS if all(c in m for m in subj)}
            out.append(row(f"**mean ({len(subj)} subjects)**", mean))
        for run, name in [("abl_no_semantic", "w/o semantic path (sub 1)"),
                          ("abl_no_geometric", "w/o geometric path (sub 1)"),
                          ("abl_no_guidance", "w/o both (sub 1)")]:
            out.append(row(name, load(args.results, run, protocol)))

        out.append("\n### Table 2: new subjects (model trained on subject 1)\n")
        out.append(header(COLS))
        out.append(row("NS-SC (subject 9)", load(args.results, "ns_sc_sub-0009", protocol)))
        out.append(row("NS-NC (subject 11)", load(args.results, "ns_nc_sub-0011", protocol)))

        out.append("\n### Table 3: number of decoded viewpoints d (subject 1)\n")
        out.append(header(COLS, "runtime (s)"))
        for d in (1, 3, 5, 9, 18):
            run = "ss_sc_sub-0001" if d == 5 else f"views{d}_sub-0001"
            out.append(row(f"d = {d}", load(args.results, run, protocol), extra=runtime(args.results, run)))

        out.append("\n### Table 4: losses (subject 1)\n")
        out.append(header(COLS))
        for run, name in [("abl_loss_latent_only", "Latent"), ("abl_loss_mse_only", "MSE"),
                          ("abl_loss_latent_mse", "Latent + MSE"), ("ss_sc_sub-0001", "Latent + MSE + Dice")]:
            out.append(row(name, load(args.results, run, protocol)))
        out.append("")

    text = "\n".join(out)
    print(text)
    with open(os.path.join(args.results, "RESULTS.md"), "w") as f:
        f.write("# Results\n\nFPD x 1e-1, CD x 1e2, EMD x 1e2.\n\n" + text)


if __name__ == "__main__":
    main()
