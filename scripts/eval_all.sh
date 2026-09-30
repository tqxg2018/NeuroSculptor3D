#!/usr/bin/env bash
# Reconstruct + evaluate everything reported in the README (run after scripts/train_all.sh).
#   FPD_CKPT=checkpoints/pointnet_fpd.pth bash scripts/eval_all.sh
set -e
cd "$(dirname "$0")/.."
FPD_CKPT=${FPD_CKPT:-checkpoints/pointnet_fpd.pth}
CKPT() { echo "outputs/$1/model_epoch200.pth"; }

evaluate() {  # <pred dir> <gt dir> <split>
    for p in paper corrected; do
        python scripts/evaluate.py --pred "$1" --gt "$2" --split "$3" --fpd_ckpt "$FPD_CKPT" --protocol $p
    done
}

# Table 1 (top): SS-SC, every subject, d = 5 viewpoints
for s in 1 2 3 4 5 6 7 8; do
    exp=ss_sc_sub-000$s
    python scripts/inference.py --ckpt "$(CKPT $exp)" --out results/$exp
    evaluate results/$exp data/gt/core_test data/splits/core_test_list.txt
done

# Table 1 (bottom) and Table 4: ablations on subject 1
for exp in abl_no_semantic abl_no_geometric abl_no_guidance abl_loss_latent_only abl_loss_mse_only abl_loss_latent_mse; do
    python scripts/inference.py --ckpt "$(CKPT $exp)" --out results/$exp
    evaluate results/$exp data/gt/core_test data/splits/core_test_list.txt
done

# Table 2: new subjects, model trained on subject 1
python scripts/inference.py --ckpt "$(CKPT ss_sc_sub-0001)" --out results/ns_sc_sub-0009 \
    --fmri_h5 data/fmri/sub-0009.hdf5 --split data/splits/apt_sub0009_list.txt --zscore self
evaluate results/ns_sc_sub-0009 data/gt/core_test data/splits/apt_sub0009_list.txt
python scripts/inference.py --ckpt "$(CKPT ss_sc_sub-0001)" --out results/ns_nc_sub-0011 \
    --fmri_h5 data/fmri/sub-0011.hdf5 --split data/splits/apact_sub0011_list.txt --zscore self
evaluate results/ns_nc_sub-0011 data/gt/apact data/splits/apact_sub0011_list.txt

# Table 3: number of decoded viewpoints d (subject 1); d = 5 is results/ss_sc_sub-0001
for d in 1 3 9 18; do
    python scripts/inference.py --ckpt "$(CKPT ss_sc_sub-0001)" --out results/views${d}_sub-0001 --num_viewpoints $d
    evaluate results/views${d}_sub-0001 data/gt/core_test data/splits/core_test_list.txt
done

python scripts/collect_results.py --results results
