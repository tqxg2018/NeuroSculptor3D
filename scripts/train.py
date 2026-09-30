"""Train NeuroSculptor3D (single stage: brain encoder + diffusion prior, hierarchical guidance).

    accelerate launch --num_processes 1 --mixed_precision fp16 scripts/train.py --config configs/ss_sc_sub01.yaml
    # override any config entry, e.g.  --set train.num_epochs=300 model.text_guidance=false
"""
import argparse
import collections
import json
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from neurosculptor3d import losses as L  # noqa: E402
from neurosculptor3d.config import load_config, to_dict  # noqa: E402
from neurosculptor3d.data import FMRIShapeDataset, read_split  # noqa: E402
from neurosculptor3d.models import NeuroSculptor3DEncoder, build_prior  # noqa: E402
from neurosculptor3d.trellis_utils import CLIPTextEncoder, load_ss_decoder, setup_trellis  # noqa: E402


class NeuroSculptor3D(nn.Module):
    """Container with the two trainable parts (names match the original checkpoints)."""

    def __init__(self, cfg):
        super().__init__()
        m, p = cfg.model, cfg.prior
        self.backbone = NeuroSculptor3DEncoder(
            num_keyvoxel=m.num_keyvoxel, num_neighbor=m.num_neighbor, hidden_dim=m.hidden_dim, num_tokens=m.num_tokens,
            n_blocks=m.n_blocks, drop=m.drop, num_views=m.num_views,
            text_guidance=m.text_guidance, structure_guidance=m.structure_guidance)
        self.diffusion_prior = build_prior(depth=p.depth, dim_head=p.dim_head, timesteps=p.timesteps,
                                           cond_drop_prob=p.cond_drop_prob,
                                           grad_checkpoint=getattr(p, "grad_checkpoint", False))


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_optimizer(model, cfg, total_steps):
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]   # (as in MindEye2; module-named norms keep decay)
    groups = []
    for part in (model.backbone, model.diffusion_prior):
        groups.append({"params": [p for n, p in part.named_parameters() if not any(nd in n for nd in no_decay)],
                       "weight_decay": cfg.train.weight_decay})
        groups.append({"params": [p for n, p in part.named_parameters() if any(nd in n for nd in no_decay)],
                       "weight_decay": 0.0})
    opt = torch.optim.AdamW(groups, lr=cfg.train.max_lr)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg.train.max_lr, total_steps=total_steps,
                                                final_div_factor=1000, last_epoch=-1,
                                                pct_start=2 / cfg.train.num_epochs)
    return opt, sched


def compute_losses(model, batch, text_target, ss_decoder, cfg, mixup, epoch_temp, stats, prefix):
    """Forward + all losses of Eq. (4)-(8). Returns the weighted total loss and accumulates logging stats."""
    lw, m = cfg.loss, cfg.model
    voxels, views = batch["voxels"], batch["view"]
    image_target, structure_target = batch["image_target"], batch.get("structure_target")

    perm = betas = select = None
    if mixup:
        voxels, perm, betas, select = L.mixco(voxels)
    out = model.backbone(voxels, views)

    loss_prior, prior_out = model.diffusion_prior(text_embed=out["image"], image_embed=image_target)
    total = lw.prior * loss_prior
    stats[f"{prefix}/loss_prior"] += loss_prior.item()
    if prefix == "train":
        stats["train/recon_cossim"] += F.cosine_similarity(prior_out, image_target).mean().item()
        stats["train/recon_mse"] += F.mse_loss(prior_out, image_target).item()

    # contrastive alignment of each path (BiMixCo during the first epochs, SoftCLIP afterwards)
    paths = [("image", out["image_proj"], image_target, lw.image_latent)]
    if m.text_guidance:
        paths.append(("text", out["text_proj"], text_target, lw.text_latent))
    if m.structure_guidance:
        paths.append(("structure", out["structure_proj"], structure_target, lw.structure_latent))
    clip_total = 0.
    for name, pred, targ, w in paths:
        pred_n = F.normalize(pred.flatten(1), dim=-1)
        targ_n = F.normalize(targ.flatten(1), dim=-1)
        if cfg.loss.use_latent:
            if mixup:
                lc = L.mixco_nce(pred_n, targ_n, temp=.006, perm=perm, betas=betas, select=select)
            else:
                lc = L.soft_clip_loss(pred_n, targ_n, temp=epoch_temp)
            stats[f"{prefix}/loss_{name}_clip_total"] += lc.item()
            clip_total += lc.item()
            total = total + w * lc
        labels = torch.arange(len(targ_n), device=targ_n.device)
        key = f"{prefix}/{'test_' if prefix == 'test' else ''}{name}"
        stats[f"{key}_fwd_pct_correct"] += L.topk(L.batchwise_cosine_similarity(targ_n, pred_n), labels, k=1).item()
        stats[f"{key}_bwd_pct_correct"] += L.topk(L.batchwise_cosine_similarity(pred_n, targ_n), labels, k=1).item()
    stats[f"{prefix}/loss_clip_total"] += clip_total

    if m.structure_guidance:
        mse = F.mse_loss(out["structure"], structure_target)
        stats[f"{prefix}/loss_structure_mse"] += mse.item()
        if cfg.loss.use_mse:
            total = total + lw.structure_mse * mse
        # Dice between the occupancies decoded (TRELLIS SS decoder) from predicted and target latents
        with torch.no_grad():
            occ_target = torch.sigmoid(ss_decoder(structure_target))
        occ_pred = torch.sigmoid(ss_decoder(out["structure"]))
        dice = L.dice_loss(occ_pred, occ_target.float())
        stats[f"{prefix}/loss_ss"] += dice.item()
        if cfg.loss.use_dice:
            total = total + lw.dice * dice
    if m.text_guidance and lw.text_mse > 0:
        tm = F.mse_loss(out["text"], text_target)
        stats[f"{prefix}/loss_text_mse"] += tm.item()
        if cfg.loss.use_mse:
            total = total + lw.text_mse * tm
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[], help="config overrides, e.g. train.max_lr=1e-4")
    ap.add_argument("--trellis_root", default=None)
    ap.add_argument("--max_steps_per_epoch", type=int, default=0, help="debug: truncate epochs")
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)

    accelerator = Accelerator(mixed_precision=cfg.train.mixed_precision)
    if accelerator.num_processes > 1:
        # the loss calls sub-modules of the unwrapped model, which would bypass DDP gradient synchronisation
        raise NotImplementedError("training is single-GPU (as for the paper); launch with --num_processes 1")
    device = accelerator.device
    seed_everything(cfg.seed)
    out_dir = os.path.join(cfg.output_dir, cfg.experiment)
    os.makedirs(out_dir, exist_ok=True)
    if accelerator.is_main_process:
        with open(os.path.join(out_dir, "config.json"), "w") as f:
            json.dump(to_dict(cfg), f, indent=2)

    # ---------------- data ----------------
    d = cfg.data
    train_ids, test_ids = read_split(d.train_split), read_split(d.test_split)
    common = dict(train_ids=train_ids, feature_root=d.feature_root, latent_root=d.latent_root,
                  num_views=d.num_views, num_keyvoxel=cfg.model.num_keyvoxel, num_neighbor=cfg.model.num_neighbor,
                  zscore=d.zscore)
    train_set = FMRIShapeDataset(d.fmri_h5, "train", train_ids, **common)
    test_set = FMRIShapeDataset(d.fmri_h5, "test", test_ids, **common)
    loader_kw = dict(batch_size=cfg.train.batch_size, num_workers=d.num_workers, pin_memory=True,
                     persistent_workers=d.num_workers > 0)
    train_loader = DataLoader(train_set, shuffle=True, drop_last=False, **loader_kw)
    test_loader = DataLoader(test_set, shuffle=False, drop_last=False, **loader_kw)
    accelerator.print(f"train {len(train_set)} / test {len(test_set)} stimuli, {train_set.num_voxels} voxels")

    # ---------------- models ----------------
    setup_trellis(args.trellis_root)
    model = NeuroSculptor3D(cfg)
    text_encoder = CLIPTextEncoder(cfg.trellis.text_encoder).to(device) if cfg.model.text_guidance else None
    ss_decoder = load_ss_decoder(cfg.trellis.image_ckpt).to(device).requires_grad_(False) \
        if cfg.model.structure_guidance else None

    steps_per_epoch = len(train_loader)
    total_steps = cfg.train.num_epochs * steps_per_epoch
    optimizer, scheduler = build_optimizer(model, cfg, total_steps)
    model, optimizer, train_loader, test_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, test_loader, scheduler)
    unwrapped = accelerator.unwrap_model(model)

    start_epoch = 0
    ckpt_path = os.path.join(out_dir, "last.pth")
    if os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        unwrapped.load_state_dict(ck["model_state_dict"])
        optimizer.load_state_dict(ck["optimizer_state_dict"])
        scheduler.load_state_dict(ck["lr_scheduler"])
        start_epoch = ck["epoch"] + 1
        accelerator.print(f"resumed from {ckpt_path} at epoch {start_epoch}")

    if cfg.wandb and accelerator.is_main_process:
        import wandb
        wandb.init(project=cfg.wandb_project, name=cfg.experiment, config=to_dict(cfg), resume="allow",
                   id=cfg.experiment.replace("/", "_"))

    n_mix = int(cfg.train.mixup_pct * cfg.train.num_epochs)
    soft_temps = L.cosine_anneal(0.004, 0.0075, cfg.train.num_epochs - n_mix)
    log_path = os.path.join(out_dir, "metrics.jsonl")

    def text_targets(batch):
        return text_encoder(list(batch["text"])) if text_encoder is not None else None

    for epoch in range(start_epoch, cfg.train.num_epochs):
        model.train()
        mixup = epoch < n_mix
        temp = None if mixup else soft_temps[epoch - n_mix].item()
        stats = collections.defaultdict(float)
        t0, losses = time.time(), []
        for batch in tqdm(train_loader, desc=f"epoch {epoch + 1}/{cfg.train.num_epochs}",
                          disable=not accelerator.is_main_process, leave=False):
            with accelerator.autocast():
                tt = text_targets(batch)
                optimizer.zero_grad()
                loss = compute_losses(unwrapped, batch, tt, ss_decoder, cfg, mixup, temp, stats, "train")
            if torch.isnan(loss):
                raise ValueError("NaN loss")
            accelerator.backward(loss)
            optimizer.step()
            scheduler.step()
            losses.append(loss.item())
            if args.max_steps_per_epoch and len(losses) >= args.max_steps_per_epoch:
                break
        n_train = len(losses)
        logs = {k: v / n_train for k, v in stats.items()}
        logs.update({"epoch": epoch, "train/loss": float(np.mean(losses)), "train/lr": optimizer.param_groups[0]["lr"],
                     "train/num_steps": (epoch + 1) * steps_per_epoch, "time/epoch_sec": time.time() - t0,
                     "gpu/max_mem_gb": torch.cuda.max_memory_allocated() / 2 ** 30})

        if accelerator.is_main_process and (epoch + 1) % cfg.train.eval_every == 0:
            model.eval()
            tstats, tlosses = collections.defaultdict(float), []
            with torch.no_grad(), accelerator.autocast():
                for batch in test_loader:
                    tt = text_targets(batch)
                    tlosses.append(compute_losses(unwrapped, batch, tt, ss_decoder, cfg, False, .006, tstats, "test").item())
            logs.update({k: v / len(test_loader) for k, v in tstats.items()})
            logs["test/loss"] = float(np.mean(tlosses))

        if accelerator.is_main_process:
            with open(log_path, "a") as f:
                f.write(json.dumps(logs) + "\n")
            if cfg.wandb:
                import wandb
                wandb.log(logs)
            state = {"epoch": epoch, "model_state_dict": unwrapped.state_dict(),
                     "optimizer_state_dict": optimizer.state_dict(), "lr_scheduler": scheduler.state_dict(),
                     "config": to_dict(cfg)}
            torch.save(state, ckpt_path + ".tmp")
            os.replace(ckpt_path + ".tmp", ckpt_path)
            if (epoch + 1) % cfg.train.save_every == 0 or epoch + 1 == cfg.train.num_epochs:
                torch.save({"epoch": epoch, "model_state_dict": unwrapped.state_dict(), "config": to_dict(cfg)},
                           os.path.join(out_dir, f"model_epoch{epoch + 1:03d}.pth"))
            accelerator.print(json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in logs.items()
                                          if k in ("epoch", "train/loss", "test/loss", "train/loss_prior",
                                                   "test/test_image_fwd_pct_correct", "time/epoch_sec")}))
    accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
