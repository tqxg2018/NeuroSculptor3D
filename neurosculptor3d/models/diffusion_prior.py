"""Diffusion prior mapping brain tokens to DINOv2 image tokens.

Adapted from MindEye2 (MIT License, MedARC): https://github.com/MedARC-AI/MindEyeV2
which builds on DALLE2-pytorch (MIT License, Phil Wang): https://github.com/lucidrains/DALLE2-pytorch
"""
import random

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from dalle2_pytorch import DiffusionPrior
from dalle2_pytorch.dalle2_pytorch import (Attention, FeedForward, LayerNorm, MLP, Rearrange, RelPosBias,
                                           RotaryEmbedding, SinusoidalPosEmb, default, exists, l2norm,
                                           prob_mask_like, rearrange, repeat)


class BrainDiffusionPrior(DiffusionPrior):
    """DiffusionPrior that is conditioned on brain tokens (passed as `text_embed`) and returns (loss, prediction)."""

    @torch.no_grad()
    def p_sample(self, x, t, text_cond=None, self_cond=None, clip_denoised=True, cond_scale=1., generator=None):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(
            x=x, t=t, text_cond=text_cond, self_cond=self_cond, clip_denoised=clip_denoised, cond_scale=cond_scale)
        noise = torch.randn_like(x)
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise, x_start

    @torch.no_grad()
    def p_sample_loop(self, *args, timesteps=None, **kwargs):
        """DDPM when `timesteps` equals the training schedule length, otherwise DDIM (eta=1) with `timesteps` steps."""
        timesteps = default(timesteps, self.noise_scheduler.num_timesteps)
        assert timesteps <= self.noise_scheduler.num_timesteps
        if timesteps < self.noise_scheduler.num_timesteps:
            return self.p_sample_loop_ddim(*args, **kwargs, timesteps=timesteps)
        return self.p_sample_loop_ddpm(*args, **kwargs)

    @torch.no_grad()
    def p_sample_loop_ddpm(self, shape, text_cond, cond_scale=1., generator=None):
        batch, device = shape[0], self.device
        image_embed = torch.randn(shape, device=device, generator=generator)
        x_start = None
        if self.init_image_embed_l2norm:
            image_embed = l2norm(image_embed) * self.image_embed_scale
        for i in reversed(range(0, self.noise_scheduler.num_timesteps)):
            times = torch.full((batch,), i, device=device, dtype=torch.long)
            self_cond = x_start if self.net.self_cond else None
            image_embed, x_start = self.p_sample(image_embed, times, text_cond=text_cond, self_cond=self_cond,
                                                 cond_scale=cond_scale)
        if self.sampling_final_clamp_l2norm and self.predict_x_start:
            image_embed = self.l2norm_clamp_embed(image_embed)
        return image_embed

    def p_losses(self, image_embed, times, text_cond, noise=None):
        noise = default(noise, lambda: torch.randn_like(image_embed))
        image_embed_noisy = self.noise_scheduler.q_sample(x_start=image_embed, t=times, noise=noise)

        self_cond = None
        if self.net.self_cond and random.random() < 0.5:
            with torch.no_grad():
                self_cond = self.net(image_embed_noisy, times, **text_cond).detach()

        pred = self.net(image_embed_noisy, times, self_cond=self_cond,
                        text_cond_drop_prob=self.text_cond_drop_prob,
                        image_cond_drop_prob=self.image_cond_drop_prob, **text_cond)
        if self.predict_x_start and self.training_clamp_l2norm:
            pred = self.l2norm_clamp_embed(pred)

        if self.predict_v:
            target = self.noise_scheduler.calculate_v(image_embed, times, noise)
        elif self.predict_x_start:
            target = image_embed
        else:
            target = noise
        return nn.functional.mse_loss(pred, target), pred

    def forward(self, text_embed=None, image_embed=None, *args, **kwargs):
        """text_embed: brain tokens (B, N, D); image_embed: target DINOv2 tokens (B, N, D). Returns (loss, x0-prediction)."""
        assert exists(text_embed) and exists(image_embed)
        times = self.noise_scheduler.sample_random_times(image_embed.shape[0])
        return self.p_losses(image_embed, times, text_cond=dict(text_embed=text_embed), *args, **kwargs)


class FlaggedCausalTransformer(nn.Module):
    def __init__(self, *, dim, depth, dim_head=64, heads=8, ff_mult=4, norm_in=False, norm_out=True,
                 attn_dropout=0., ff_dropout=0., final_proj=True, normformer=False, rotary_emb=True, causal=True,
                 grad_checkpoint=False):
        super().__init__()
        # activation checkpointing: with ~2.7k tokens the attention maps dominate GPU memory (~40 GB at batch 4);
        # recomputing each block in the backward pass is exact and cuts this to a few GB
        self.grad_checkpoint = grad_checkpoint
        self.init_norm = LayerNorm(dim) if norm_in else nn.Identity()
        self.rel_pos_bias = RelPosBias(heads=heads)
        rotary_emb = RotaryEmbedding(dim=min(32, dim_head)) if rotary_emb else None
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim=dim, causal=causal, dim_head=dim_head, heads=heads, dropout=attn_dropout, rotary_emb=rotary_emb),
                FeedForward(dim=dim, mult=ff_mult, dropout=ff_dropout, post_activation_norm=normformer),
            ]))
        self.norm = LayerNorm(dim, stable=True) if norm_out else nn.Identity()
        self.project_out = nn.Linear(dim, dim, bias=False) if final_proj else nn.Identity()

    def forward(self, x):
        n, device = x.shape[1], x.device
        x = self.init_norm(x)
        attn_bias = self.rel_pos_bias(n, n + 1, device=device)
        for attn, ff in self.layers:
            if self.grad_checkpoint and self.training and torch.is_grad_enabled():
                x = checkpoint(self._block, attn, ff, x, attn_bias, use_reentrant=False)
            else:
                x = self._block(attn, ff, x, attn_bias)
        return self.project_out(self.norm(x))

    @staticmethod
    def _block(attn, ff, x, attn_bias):
        x = attn(x, attn_bias=attn_bias) + x
        return ff(x) + x


class PriorNetwork(nn.Module):
    """Transformer denoiser over [brain tokens | time token | noisy image tokens (+ learned positional embedding)]."""

    def __init__(self, dim, num_timesteps=None, num_time_embeds=1, num_tokens=257, causal=True,
                 learned_query_mode="none", **kwargs):
        super().__init__()
        self.dim = dim
        self.num_time_embeds = num_time_embeds
        self.continuous_embedded_time = not exists(num_timesteps)
        self.learned_query_mode = learned_query_mode

        self.to_time_embeds = nn.Sequential(
            nn.Embedding(num_timesteps, dim * num_time_embeds) if exists(num_timesteps)
            else nn.Sequential(SinusoidalPosEmb(dim), MLP(dim, dim * num_time_embeds)),
            Rearrange("b (n d) -> b n d", n=num_time_embeds),
        )
        if learned_query_mode == "token":
            self.learned_query = nn.Parameter(torch.randn(num_tokens, dim))
        if learned_query_mode == "pos_emb":
            self.learned_query = nn.Parameter(torch.randn(num_tokens, dim) * dim ** -0.5)
        if learned_query_mode == "all_pos_emb":
            self.learned_query = nn.Parameter(torch.randn(num_tokens * 2 + 1, dim) * dim ** -0.5)
        self.causal_transformer = FlaggedCausalTransformer(dim=dim, causal=causal, **kwargs)

        self.null_brain_embeds = nn.Parameter(torch.randn(num_tokens, dim))
        self.null_image_embed = nn.Parameter(torch.randn(num_tokens, dim))
        self.num_tokens = num_tokens
        self.self_cond = False

    def forward_with_cond_scale(self, *args, cond_scale=1., **kwargs):
        logits = self.forward(*args, **kwargs)
        if cond_scale == 1:
            return logits
        null_logits = self.forward(*args, brain_cond_drop_prob=1., image_cond_drop_prob=1, **kwargs)
        return null_logits + (logits - null_logits) * cond_scale

    def forward(self, image_embed, diffusion_timesteps, *, self_cond=None, brain_embed=None, text_embed=None,
                brain_cond_drop_prob=0., text_cond_drop_prob=None, image_cond_drop_prob=0.):
        if text_embed is not None:
            brain_embed = text_embed
        if text_cond_drop_prob is not None:
            brain_cond_drop_prob = text_cond_drop_prob

        batch, _, dim, device, dtype = *image_embed.shape, image_embed.device, image_embed.dtype

        # classifier-free guidance masks
        brain_keep_mask = rearrange(prob_mask_like((batch,), 1 - brain_cond_drop_prob, device=device), "b -> b 1 1")
        image_keep_mask = rearrange(prob_mask_like((batch,), 1 - image_cond_drop_prob, device=device), "b -> b 1 1")
        brain_embed = torch.where(brain_keep_mask, brain_embed, self.null_brain_embeds.to(brain_embed.dtype)[None])
        image_embed = torch.where(image_keep_mask, image_embed, self.null_image_embed.to(image_embed.dtype)[None])

        if self.continuous_embedded_time:
            diffusion_timesteps = diffusion_timesteps.type(dtype)
        time_embed = self.to_time_embeds(diffusion_timesteps)

        if self.learned_query_mode == "token":
            learned_queries = repeat(self.learned_query, "n d -> b n d", b=batch)
        elif self.learned_query_mode == "pos_emb":
            image_embed = image_embed + repeat(self.learned_query, "n d -> b n d", b=batch)
            learned_queries = torch.empty((batch, 0, dim), device=brain_embed.device)
        else:
            learned_queries = torch.empty((batch, 0, dim), device=brain_embed.device)

        tokens = torch.cat((brain_embed, time_embed, image_embed, learned_queries), dim=-2)
        if self.learned_query_mode == "all_pos_emb":
            tokens = tokens + repeat(self.learned_query, "n d -> b n d", b=batch)

        tokens = self.causal_transformer(tokens)
        return tokens[..., -self.num_tokens:, :]


def build_prior(num_tokens: int = 1374, dim: int = 1024, depth: int = 6, dim_head: int = 52, timesteps: int = 100,
                cond_drop_prob: float = 0.2, grad_checkpoint: bool = False) -> BrainDiffusionPrior:
    """Prior configuration used in the paper (MindEye2 defaults: depth 6, 52-dim heads, 100 cosine timesteps)."""
    net = PriorNetwork(dim=dim, depth=depth, dim_head=dim_head, heads=dim // dim_head, causal=False,
                       num_tokens=num_tokens, learned_query_mode="pos_emb", grad_checkpoint=grad_checkpoint)
    return BrainDiffusionPrior(net=net, image_embed_dim=dim, condition_on_text_encodings=False, timesteps=timesteps,
                               cond_drop_prob=cond_drop_prob, image_embed_scale=None)
