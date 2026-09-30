"""Image-level metrics on rendered views: n-way top-1 accuracy, LPIPS, SSIM.

n-way top-1 follows MinD-Video / MinD-3D: an ImageNet classifier (torchvision ViT-H/14) labels the ground-truth
view; the reconstruction is correct if, among the probability of that label and of n-1 randomly drawn other
labels, the true label has the highest probability. Averaged over `trials` random draws and all views.
"""
from typing import List, Sequence

import numpy as np
import torch
from PIL import Image
from torchvision import transforms


class NWayTop1:
    def __init__(self, device="cuda", batch_size=16):
        from torchvision.models import ViT_H_14_Weights, vit_h_14
        weights = ViT_H_14_Weights.DEFAULT
        self.model = vit_h_14(weights=weights).to(device).eval()
        self.preprocess = weights.transforms()
        self.device, self.batch_size = device, batch_size

    @torch.no_grad()
    def probs(self, images: Sequence[np.ndarray]) -> torch.Tensor:
        out = []
        for i in range(0, len(images), self.batch_size):
            x = torch.stack([self.preprocess(Image.fromarray(im)) for im in images[i:i + self.batch_size]]).to(self.device)
            out.append(self.model(x).softmax(-1).cpu())
        return torch.cat(out)

    @staticmethod
    def accuracy(pred_probs: torch.Tensor, gt_ids: np.ndarray, n_way: int, trials: int = 100, seed: int = 0) -> List[float]:
        """Per-image accuracy averaged over `trials` random draws of n-1 distractor classes."""
        rng = np.random.RandomState(seed)
        pred_probs = pred_probs.numpy()
        num_classes = pred_probs.shape[1]
        accs = []
        for prob, cid in zip(pred_probs, gt_ids):
            others = np.delete(np.arange(num_classes), cid)
            picks = np.stack([rng.choice(others, n_way - 1, replace=False) for _ in range(trials)])
            accs.append(float((prob[cid] >= prob[picks].max(1)).mean()))
        return accs


class LPIPSMetric:
    """AlexNet LPIPS. `input_range='01'` feeds [0,1] images directly (the protocol used for the paper numbers);
    `input_range='m11'` rescales to [-1,1] as the LPIPS package expects."""

    def __init__(self, device="cuda", input_range="01"):
        import lpips
        assert input_range in ("01", "m11")
        self.fn = lpips.LPIPS(net="alex", verbose=False).to(device)
        self.device, self.input_range = device, input_range

    @torch.no_grad()
    def __call__(self, preds: Sequence[np.ndarray], gts: Sequence[np.ndarray], batch_size=64) -> List[float]:
        to_t = transforms.ToTensor()
        out = []
        for i in range(0, len(preds), batch_size):
            P = torch.stack([to_t(x) for x in preds[i:i + batch_size]]).to(self.device)
            G = torch.stack([to_t(x) for x in gts[i:i + batch_size]]).to(self.device)
            out.append(self.fn(G, P, normalize=self.input_range == "m11").flatten().cpu())
        return torch.cat(out).tolist()


def ssim_scores(preds: Sequence[np.ndarray], gts: Sequence[np.ndarray]) -> List[float]:
    """SSIM as in MindEye: resize to 256 then 425 (bilinear), grayscale, Gaussian-weighted SSIM."""
    from skimage.color import rgb2gray
    from skimage.metrics import structural_similarity
    to_t = transforms.ToTensor()
    r256 = transforms.Resize((256, 256))
    r425 = transforms.Resize(425, interpolation=transforms.InterpolationMode.BILINEAR)
    scores = []
    for p, g in zip(preds, gts):
        P = rgb2gray(r425(r256(to_t(p)[None])).permute(0, 2, 3, 1).numpy()[0])
        G = rgb2gray(r425(r256(to_t(g)[None])).permute(0, 2, 3, 1).numpy()[0])
        scores.append(structural_similarity(P, G, gaussian_weights=True, sigma=1.5, use_sample_covariance=False,
                                            data_range=1.0))
    return scores
