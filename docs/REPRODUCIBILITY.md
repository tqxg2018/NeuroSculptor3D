# Reproducibility notes

This repository is a cleaned-up re-implementation of the research code used for the paper. The released checkpoints
were retrained with this code, and all numbers in the README were produced with it. They differ from the numbers in
the paper, which were obtained with an earlier version of the code, data and evaluation scripts.

## Training configuration

The model, losses and optimisation follow the configuration of the original subject-1 checkpoint (recovered from its
state dict and training logs):

| | released config | text of the paper |
|---|---|---|
| epochs | 200 | 300 |
| batch size / max. LR / schedule | 4 / 3e-5 / one-cycle | same |
| loss weights | prior 30, contrastive: image 10, text 1, structure 1; MSE 1e4; Dice 2 | lambda_Latent = 1 for all paths |
| semantic path | contrastive loss only | contrastive + MSE (Eq. 5) |
| semantic target | CLIP ViT-L/14 (OpenAI weights, as in TRELLIS) tokens of the category name, e.g. "car" | OpenCLIP, "A 3D model of a car." (Fig. 2) |
| viewpoint embedding | appended as an extra brain token, then a linear map over the token axis | concatenated along the feature dimension (Eq. 2) |
| mixer blocks / viewpoints | 1 / 32 | - |

A semantic-path MSE term is available via `loss.text_mse` (disabled by default).

## fMRI data

All subjects use one consistent run of the preprocessing pipeline (the files released on Hugging Face). The original
subject-1 checkpoint used an earlier run for subject 1 only; in linear-probe tests (13-way category decoding from
single-trial training data) the released version carries more decodable signal (0.57 vs 0.42 test accuracy).

## Verification of the re-implementation

* **Model.** With the original subject-1 checkpoint, the re-implemented encoder and diffusion prior give bit-identical
  outputs, training losses and (seeded) DDIM samples to the original code.
* **Training.** Retraining with the original configuration and data reproduces the per-epoch training and test losses
  of the original run (e.g. epoch 22: total loss 1166 vs 1165, structure MSE 0.1153 vs 0.1152).
* **Data.** Frame extraction is reproduced bit-exactly; DINOv2 targets bit-exactly from the released preprocessed
  frames (also in a freshly installed environment); sparse-structure latents up to fp16 noise.
* **Evaluation.** The ground-truth views are reproduced bit-exactly for 64/104 test objects and with small texture
  differences for the rest, which leave the metrics unchanged (+-0.001).

Engineering changes that do not alter the method: only the sampled frame's DINOv2 tokens are read per training
sample (the original loader read all 32 frames, ~3x slower training), activation checkpointing of the prior
(bit-exact gradients, ~18 GB instead of ~40 GB GPU memory), per-object seeding at inference, and TRELLIS is used
unmodified.
