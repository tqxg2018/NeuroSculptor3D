"""Training losses (BiMixCo / SoftCLIP from MindEye2, Dice for voxel occupancy)."""
import torch
import torch.nn.functional as F


def mixco(voxels, beta=0.15, s_thresh=0.5, perm=None, betas=None, select=None):
    """BiMixCo input mixing (MindEye). Mixes a random half of the batch with a shuffled copy."""
    if perm is None:
        perm = torch.randperm(voxels.shape[0])
    voxels_shuffle = voxels[perm].to(voxels.device, dtype=voxels.dtype)
    if betas is None:
        betas = torch.distributions.Beta(beta, beta).sample([voxels.shape[0]]).to(voxels.device, dtype=voxels.dtype)
    if select is None:
        select = (torch.rand(voxels.shape[0]) <= s_thresh).to(voxels.device)
    betas_shape = [-1] + [1] * (len(voxels.shape) - 1)
    voxels[select] = voxels[select] * betas[select].reshape(*betas_shape) + \
        voxels_shuffle[select] * (1 - betas[select]).reshape(*betas_shape)
    betas[~select] = 1
    return voxels, perm, betas, select


def mixco_nce(preds, targs, temp=0.1, perm=None, betas=None, select=None, bidirectional=True):
    """Contrastive loss with soft labels induced by the mixing coefficients (BiMixCo)."""
    brain_clip = (preds @ targs.T) / temp
    if perm is not None and betas is not None and select is not None:
        probs = torch.diag(betas)
        probs[torch.arange(preds.shape[0]).to(preds.device), perm] = 1 - betas
        loss = -(brain_clip.log_softmax(-1) * probs).sum(-1).mean()
        if bidirectional:
            loss2 = -(brain_clip.T.log_softmax(-1) * probs.T).sum(-1).mean()
            loss = (loss + loss2) / 2
        return loss
    labels = torch.arange(brain_clip.shape[0]).to(brain_clip.device)
    loss = F.cross_entropy(brain_clip, labels)
    if bidirectional:
        loss = (loss + F.cross_entropy(brain_clip.T, labels)) / 2
    return loss


def soft_clip_loss(preds, targs, temp=0.125):
    """SoftCLIP (MindEye): cross-entropy against the target-target similarity distribution."""
    clip_clip = (targs @ targs.T) / temp
    brain_clip = (preds @ targs.T) / temp
    loss1 = -(brain_clip.log_softmax(-1) * clip_clip.softmax(-1)).sum(-1).mean()
    loss2 = -(brain_clip.T.log_softmax(-1) * clip_clip.softmax(-1)).sum(-1).mean()
    return (loss1 + loss2) / 2


def dice_loss(pred_prob, target_prob, eps=1.0):
    """Soft Dice over the whole batch (as in the paper code: sums over all voxels of all samples)."""
    inter = (pred_prob * target_prob).sum()
    return 1 - (2 * inter + eps) / (pred_prob.sum() + target_prob.sum() + eps)


def cosine_anneal(start, end, steps):
    return end + (start - end) / 2 * (1 + torch.cos(torch.pi * torch.arange(steps) / (steps - 1)))


def batchwise_cosine_similarity(Z, B):
    Z = Z.flatten(1)
    B = B.flatten(1).T
    Z_norm = torch.linalg.norm(Z, dim=1, keepdim=True)
    B_norm = torch.linalg.norm(B, dim=0, keepdim=True)
    return ((Z @ B) / (Z_norm @ B_norm)).T


def topk(similarities, labels, k=5):
    if k > similarities.shape[0]:
        k = similarities.shape[0]
    topsum = 0
    for i in range(k):
        topsum += torch.sum(torch.argsort(similarities, axis=1)[:, -(i + 1)] == labels) / len(labels)
    return topsum
