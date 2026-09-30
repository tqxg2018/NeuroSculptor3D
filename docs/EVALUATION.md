# Evaluation

`scripts/evaluate.py` compares the reconstructions written by `scripts/inference.py` with ground truth prepared by
`scripts/data/prepare_gt.py`.

## Settings

| Setting | Model trained on | Evaluated on | Objects |
|---|---|---|---|
| SS-SC | subject s (1-8) | subject s | 104 core-test objects (13 categories), 2 presentations averaged |
| NS-SC | subject 1 | subject 9 | the same 104 objects, 2 presentations averaged |
| NS-NC | subject 1 | subject 11 | 220 objects of 55 categories (42 unseen), repeated presentations averaged |

## Metrics

**Image level**, on six views per object (a 300-frame turntable, frames 0, 60, 120, 179, 239, 299):

* *n-way top-1* (n = 2, 10): an ImageNet classifier (torchvision ViT-H/14, SWAG) labels the ground-truth view; a
  reconstruction is correct if the probability of that label exceeds those of n-1 random other labels
  (100 random draws, averaged over all views and objects).
* *LPIPS* (AlexNet) and *SSIM* (grayscale, Gaussian-weighted, MindEye resizing).

Reconstructions are rendered with the TRELLIS Gaussian renderer (radius 2, 40 deg FoV, oscillating pitch, black
background); ground truth with pyrender (textured ShapeNet model, camera distance 2.5, 60 deg FoV, black background).

**Structure level**, on 2,048 points: *FPD* (Frechet distance of PointNet features, 13-class ShapeNet
classifier from the MinD-3D evaluation), *Chamfer distance* (sum of both mean squared nearest-neighbour distances)
and *EMD* (approximate, Fan et al.). The prediction is the set of 3D-Gaussian centres, the ground truth the
surface of the normalised ShapeNet mesh. Tables report FPD x 1e-1, CD x 1e2 and EMD x 1e2 (MinD-3D convention).

## Protocols

| | `--protocol paper` (default) | `--protocol corrected` |
|---|---|---|
| LPIPS input | images in [0, 1] | images in [-1, 1] (as LPIPS expects) |
| point-cloud frames | prediction y-up, ground truth z-up | ground truth rotated to y-up |
| point-cloud normalisation | none | both centred and scaled into a sphere of radius 0.5 |

`paper` reproduces the evaluation that was used for the numbers in the paper. `corrected` removes the frame mismatch
and scale sensitivity of the structural metrics and uses LPIPS as intended; we report both.

FPD is computed from only 104 (or 220) point clouds with 1,805-d features, so its covariance estimate is rank-deficient
and the value varies noticeably with the random point sampling; CD/EMD and the image metrics are stable.
