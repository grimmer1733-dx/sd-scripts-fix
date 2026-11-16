"""Loss / noise scheduling helpers extracted from ``library.train_util``.

This module hosts the per-step loss building blocks shared by every
trainer:

- :func:`get_timesteps` — sample diffusion timesteps for a batch.
- :func:`get_noise_noisy_latents_and_timesteps` — sample noise + timestep
  and produce noisy latents (with noise-offset / multires-noise / IP-noise
  variants applied).
- :func:`get_huber_threshold_if_needed` — per-timestep Huber/Smooth-L1
  threshold schedule (exponential / SNR / constant).
- :func:`conditional_loss` — dispatch over ``l2 / l1 / huber / smooth_l1``.

These used to live in ``library.train_util`` and are still re-exported
from there for backward compatibility. New code should import from this
module.
"""

import math
import os
from typing import Optional, Tuple

import torch

from library import custom_train_functions


def get_timesteps(min_timestep: int, max_timestep: int, b_size: int, device: torch.device) -> torch.Tensor:
    if min_timestep < max_timestep:
        sampling = os.environ.get("FT_TIMESTEP_SAMPLING", "uniform").strip().lower()
        if sampling == "stratified_uniform":
            # Stratify the synchronized global DDP batch, including local batch=1 on 2 GPUs.
            world_size = max(1, int(os.environ.get("WORLD_SIZE", "1")))
            rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
            rank = min(max(rank, 0), world_size - 1)
            global_b_size = b_size * world_size
            if global_b_size > 1:
                width = max_timestep - min_timestep
                offsets = rank * b_size + torch.arange(b_size, dtype=torch.float64, device="cpu")
                jitter = torch.rand(b_size, dtype=torch.float64, device="cpu")
                timesteps = min_timestep + torch.floor((offsets + jitter) * (width / global_b_size)).long()
                timesteps = timesteps[torch.randperm(b_size, device="cpu")]
            else:
                timesteps = torch.randint(min_timestep, max_timestep, (b_size,), device="cpu")
        elif sampling in ("uniform", ""):
            timesteps = torch.randint(min_timestep, max_timestep, (b_size,), device="cpu")
        else:
            raise ValueError(f"Unknown FT_TIMESTEP_SAMPLING={sampling!r}")
    else:
        timesteps = torch.full((b_size,), max_timestep, device="cpu")
    return timesteps.long().to(device)


def get_noise_noisy_latents_and_timesteps(
    args, noise_scheduler, latents: torch.FloatTensor
) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.IntTensor]:
    # Sample noise that we'll add to the latents
    noise = torch.randn_like(latents, device=latents.device)
    if args.noise_offset:
        if args.noise_offset_random_strength:
            noise_offset = torch.rand(1, device=latents.device) * args.noise_offset
        else:
            noise_offset = args.noise_offset
        noise = custom_train_functions.apply_noise_offset(latents, noise, noise_offset, args.adaptive_noise_scale)
    if args.multires_noise_iterations:
        noise = custom_train_functions.pyramid_noise_like(
            noise, latents.device, args.multires_noise_iterations, args.multires_noise_discount
        )

    # Sample a random timestep for each image
    b_size = latents.shape[0]
    min_timestep = 0 if args.min_timestep is None else args.min_timestep
    max_timestep = noise_scheduler.config.num_train_timesteps if args.max_timestep is None else args.max_timestep
    timesteps = get_timesteps(min_timestep, max_timestep, b_size, latents.device)

    # Add noise to the latents according to the noise magnitude at each timestep
    # (this is the forward diffusion process)
    if args.ip_noise_gamma:
        if args.ip_noise_gamma_random_strength:
            strength = torch.rand(1, device=latents.device) * args.ip_noise_gamma
        else:
            strength = args.ip_noise_gamma
        if os.environ.get("FT_IP_NOISE_SCHEDULE", "constant").strip().lower() == "mid_snr_bridge":
            _a_device = noise_scheduler.alphas_cumprod.device
            _a_bar = torch.index_select(noise_scheduler.alphas_cumprod, 0, timesteps.to(_a_device)).to(device=latents.device, dtype=torch.float32).clamp(0.0, 1.0)
            _strength = torch.as_tensor(strength, device=latents.device, dtype=torch.float32).reshape(-1)
            if _strength.numel() == 1:
                _strength = _strength.expand(latents.shape[0])
            _strength = _strength * (2.0 * torch.sqrt(_a_bar) * torch.sqrt((1.0 - _a_bar).clamp_min(0.0)))
            strength = _strength.view(-1, 1, 1, 1).to(dtype=latents.dtype)
        noisy_latents = noise_scheduler.add_noise(latents, noise + strength * torch.randn_like(latents), timesteps)
    else:
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

    # This moves the alphas_cumprod back to the CPU after it is moved in noise_scheduler.add_noise
    noise_scheduler.alphas_cumprod = noise_scheduler.alphas_cumprod.cpu()

    return noise, noisy_latents, timesteps


def get_huber_threshold_if_needed(args, timesteps: torch.Tensor, noise_scheduler) -> Optional[torch.Tensor]:
    if not (args.loss_type == "huber" or args.loss_type == "smooth_l1"):
        return None

    b_size = timesteps.shape[0]
    if args.huber_schedule == "exponential":
        alpha = -math.log(args.huber_c) / noise_scheduler.config.num_train_timesteps
        result = torch.exp(-alpha * timesteps) * args.huber_scale
    elif args.huber_schedule == "snr":
        if not hasattr(noise_scheduler, "alphas_cumprod"):
            raise NotImplementedError("Huber schedule 'snr' is not supported with the current model.")
        alphas_cumprod = torch.index_select(noise_scheduler.alphas_cumprod, 0, timesteps.to(noise_scheduler.alphas_cumprod.device))
        sigmas = ((1.0 - alphas_cumprod) / alphas_cumprod) ** 0.5
        result = (1 - args.huber_c) / (1 + sigmas) ** 2 + args.huber_c
        result = result.to(timesteps.device)
    elif args.huber_schedule == "constant":
        result = torch.full((b_size,), args.huber_c * args.huber_scale, device=timesteps.device)
    else:
        raise NotImplementedError(f"Unknown Huber loss schedule {args.huber_schedule}!")

    return result


def conditional_loss(
    model_pred: torch.Tensor, target: torch.Tensor, loss_type: str, reduction: str, huber_c: Optional[torch.Tensor] = None
):
    """
    NOTE: if you're using the scheduled version, huber_c has to depend on the timesteps already
    """
    if loss_type == "l2":
        loss = torch.nn.functional.mse_loss(model_pred, target, reduction=reduction)
    elif loss_type == "l1":
        loss = torch.nn.functional.l1_loss(model_pred, target, reduction=reduction)
    elif loss_type == "huber":
        if huber_c is None:
            raise NotImplementedError("huber_c not implemented correctly")
        # Reshape huber_c to broadcast with model_pred (supports 4D and 5D tensors)
        huber_c = huber_c.view(-1, *([1] * (model_pred.ndim - 1)))
        loss = 2 * huber_c * (torch.sqrt((model_pred - target) ** 2 + huber_c**2) - huber_c)
        if reduction == "mean":
            loss = torch.mean(loss)
        elif reduction == "sum":
            loss = torch.sum(loss)
    elif loss_type == "smooth_l1":
        if huber_c is None:
            raise NotImplementedError("huber_c not implemented correctly")
        # Reshape huber_c to broadcast with model_pred (supports 4D and 5D tensors)
        huber_c = huber_c.view(-1, *([1] * (model_pred.ndim - 1)))
        loss = 2 * (torch.sqrt((model_pred - target) ** 2 + huber_c**2) - huber_c)
        if reduction == "mean":
            loss = torch.mean(loss)
        elif reduction == "sum":
            loss = torch.sum(loss)
    else:
        raise NotImplementedError(f"Unsupported Loss Type: {loss_type}")
    return loss
