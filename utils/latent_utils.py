from pathlib import Path
from typing import Tuple
import pathlib

import numpy as np
import torch
from PIL import Image

from cactif_model import CACTIFModel
from config import RunConfig
from utils import image_utils
from utils.ddpm_inversion import invert


def load_or_invert_one_image(
    model: CACTIFModel, cfg: RunConfig, img_path: pathlib.Path, type_img: str
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Loads latent and noise tensors from disk if available, otherwise inverts the image using DDPM inversion.

    Args:
        model: The CACTIF model.
        cfg: Model configuration.
        img_path: Path to the input image.
        type_img: Image type ("style" or "content").

    Returns:
        Tuple of (latents, noise)
    """
    if type_img not in {"style", "content"}:
        return None, None

    latent_path = cfg.style_latent_save_path if type_img == "style" else cfg.content_latent_save_path

    if cfg.load_latents and latent_path.exists():
        print("Loading existing latents...")
        return load_one_latent_noise(latent_path=latent_path)

    print("Inverting image...")
    return invert_one_image(model, cfg, img_path, type_img)


def invert_one_image(
    model: CACTIFModel, cfg: RunConfig, img_path: pathlib.Path, type_img: str
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Inverts an input image into latent space using DDPM inversion.

    Args:
        model: The CACTIF model.
        cfg: Model configuration.
        img_path: Path to the input image.
        type_img: Image type ("style" or "content").

    Returns:
        Tuple of (latents, noise)
    """
    image = image_utils.load_size(img_path)

    # Save resized image for debugging or visualization
    if cfg.output_path is not None:
        cfg.output_path.mkdir(parents=True, exist_ok=True)
        Image.fromarray(image).save(cfg.output_path / f"{type_img}.png")

    model.enable_edit = False

    # Normalize image to [-1, 1] and rearrange dimensions
    input_image = torch.from_numpy(np.array(image)).float() / 127.5 - 1.0
    input_tensor = input_image.permute(2, 0, 1).unsqueeze(0).to('cuda')

    noise, latents = invert(
        x0=input_tensor,
        pipe=model,
        prompt_src=cfg.prompt,
        num_diffusion_steps=cfg.num_timesteps,
        cfg_scale_src=3.5
    )

    model.enable_edit = True

    save_path = cfg.latents_path / type_img
    save_path.mkdir(parents=True, exist_ok=True)

    torch.save(latents, save_path / f"{img_path.stem}.pt")
    torch.save(noise, save_path / f"{img_path.stem}_ddpm_noise.pt")

    return latents, noise


def load_one_latent_noise(latent_path: Path) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Loads a latent tensor and its associated DDPM noise tensor from disk.

    Args:
        latent_path: Path to the latent .pt file.

    Returns:
        Tuple of (latents, noise)
    """
    latents = torch.load(latent_path)
    latents = [l.to("cuda") for l in latents] if isinstance(latents, list) else latents.to("cuda")

    noise_path = latent_path.parent / (latent_path.stem + "_ddpm_noise.pt")
    noise = torch.load(noise_path).to("cuda")

    return latents, noise


def get_init_latents_and_noises(model: CACTIFModel, cfg: RunConfig) -> Tuple[torch.Tensor, list]:
    """
    Assembles initial latents and noises from the model's stored values.

    Returns:
        A tuple of:
            - init_latents: Tensor of shape [3, C, H, W]
            - init_zs: List of noise tensors from DDPM inversion
    """
    # Reduce multi-step latent history if necessary
    if model.latents_content.dim() == 4 and model.latents_style.dim() == 4 and model.latents_style.shape[0] > 1:
        model.latents_content = model.latents_content[cfg.skip_steps]
        model.latents_style = model.latents_style[cfg.skip_steps]

    init_latents = torch.stack([
        model.latents_content, # transfer
        model.latents_style, # style
        model.latents_content  # content
    ])

    init_zs = [
        model.zs_content[cfg.skip_steps:],
        model.zs_style[cfg.skip_steps:],
        model.zs_content[cfg.skip_steps:]
    ]

    return init_latents, init_zs
