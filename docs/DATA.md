# Data

NeuroSculptor3D is trained and evaluated on [fMRI-Shape](https://huggingface.co/datasets/Fudan-fMRI/fMRI-Shape)
(Gao et al., MinD-3D, ECCV 2024; Apache-2.0): participants watched 8-second videos of ShapeNet objects rotating
360 degrees. We provide the preprocessed data (and our checkpoints, in `checkpoints/`) on Hugging Face at
[tqxg2022/NeuroSculptor3D-data](https://huggingface.co/datasets/tqxg2022/NeuroSculptor3D-data); this page describes the
file formats and how the derived files (frames, targets, evaluation ground truth) are produced.

## Released files

| Path | Content | Size |
|---|---|---|
| `fmri/sub-00XX.hdf5` | single-trial fMRI responses of subjects 1-8, 9, 11-13 | ~1 GB |
| `splits/core_train_list.txt`, `core_test_list.txt` | 1,300 / 104 objects of the core set (13 categories) | |
| `splits/apt_sub0009_list.txt` | the 104 core-test objects seen by subject 9 (NS-SC) | |
| `splits/apact_sub0011_list.txt` | the 220 objects (55 categories) seen by subject 11 (NS-NC) | |
| `frames/<cat>/<id>/<v>.jpg` | 32 frames (224x224) of every stimulus video, v = viewpoint id | ~0.4 GB |
| `frames_518/<cat>/<id>/<v>.png` | the frames after TRELLIS preprocessing (background removed, cropped, 518x518) | ~3.0 GB |
| `ss_latents/<cat>/<id>/latent.npz` | TRELLIS sparse-structure latent (`mean`, 8x16x16x16) | ~0.2 GB |

Objects are identified by `<ShapeNet synset>/<ShapeNet model id>`, e.g. `02691156/ed35478403ae873943cf31d2bcc8f4`.

### fMRI files

Each HDF5 file contains two datasets:

* `vox` - `(num_trials, num_voxels)` float32, one response pattern (GLMsingle beta) per stimulus presentation;
  `num_voxels` differs between subjects (17,842 - 22,525).
* `stimuli` - `(num_trials,)` bytes, the presented video, e.g. `b"./stimuli/02691156/<id>.mp4"`.

Subjects 1-8 saw the 1,300 training objects once and the 104 test objects twice; subject 9 saw the test objects
twice; subjects 11-13 saw 220 objects of 55 categories (55 of them three times). The model z-scores each voxel with
the statistics of the subject's training trials (`zscore: train`); subjects without training trials use their own
statistics (`--zscore self`). Repeated presentations of a test object are averaged.

Preprocessing (from the raw fMRI-Shape volumes): fMRIPrep in volumetric space; the union of the eight NSD
`nsdgeneral` masks registered to each participant with ANTs; GLMsingle single-trial betas within the mask. The
fMRI files are released already preprocessed.

### Voxel grouping

The variable-length voxel vector is mapped to a fixed `(G=512, K=32)` array (UniBrain): 512 anchors are spread
uniformly over the voxel order and each group holds the 32 consecutive voxels around its anchor
(`neurosculptor3d/data/fmri_shape.py:group_voxels`).

## Derived targets

**Frames** (`scripts/data/extract_frames.py`): 32 frames at `np.linspace(0, 191, 32)` of each 192-frame video,
resized to 224x224. Frame index = viewpoint id used by the viewpoint embedding.

**DINOv2 tokens** (`scripts/data/extract_dino_features.py`): each frame is preprocessed by TRELLIS
(`rembg`/u2net background removal, crop, 518x518; `scripts/data/preprocess_frames.py`) and encoded by DINOv2
ViT-L/14-reg into the 1,374 x 1,024 layer-normalised tokens that condition TRELLIS. These are the perceptual-path
targets. Computing them from the released `frames_518` (`--preprocessed`) reproduces our targets bit-exactly;
starting from the JPEG frames gives slightly different targets in most environments, because JPEG decoding differs
between Pillow builds (IJG libjpeg vs libjpeg-turbo) and propagates through the background removal.

**Sparse-structure latents** (`scripts/data/encode_ss_latents.py`): the ShapeNet mesh is normalised as in the TRELLIS
data toolkit (y-up to z-up, longest side 1, centred), voxelised at 64^3 and encoded by the TRELLIS sparse-structure VAE
(posterior mean). These are the geometric-path targets; re-encoding matches the released latents up to fp16 noise.

## Evaluation ground truth

ShapeNet models may not be redistributed, so the ground truth is rendered locally from ShapeNetCore v2 with
`scripts/data/prepare_gt.py` (six views per object and the normalised mesh). See [EVALUATION.md](EVALUATION.md).
