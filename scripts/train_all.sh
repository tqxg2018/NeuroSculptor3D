#!/usr/bin/env bash
# All training runs used for the result tables (one GPU, sequential). Each run resumes from outputs/<exp>/last.pth.
set -e
cd "$(dirname "$0")/.."
train() { accelerate launch --num_processes 1 --mixed_precision fp16 scripts/train.py \
              --config configs/neurosculptor3d.yaml --set experiment="$1" "${@:2}"; }

# Table 1 / Table 2: one SS-SC model per subject (the subject-1 model is also used for NS-SC and NS-NC)
for s in 1 2 3 4 5 6 7 8; do
    train ss_sc_sub-000$s data.fmri_h5=data/fmri/sub-000$s.hdf5
done

# Table 1 (bottom): hierarchical guidance ablation, subject 1
train abl_no_semantic  model.text_guidance=false
train abl_no_geometric model.structure_guidance=false
train abl_no_guidance  model.text_guidance=false model.structure_guidance=false

# Table 4: loss ablation, subject 1  (full model = ss_sc_sub-0001)
train abl_loss_latent_only loss.use_mse=false loss.use_dice=false
train abl_loss_mse_only    loss.use_latent=false loss.use_dice=false
train abl_loss_latent_mse  loss.use_dice=false
