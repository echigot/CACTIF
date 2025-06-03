import os
import sys
import random
import json

import numpy as np
import pyrallis
import torch
from PIL import Image
from tqdm import tqdm
from diffusers.training_utils import set_seed

sys.path.append(".")
sys.path.append("..")

from cactif_model import CACTIFModel
from config import RunConfig, Range
from utils.latent_utils import load_or_invert_one_image, get_init_latents_and_noises
from utils.mask_utils import *
from utils.image_utils import open_style_labels
import palettes


@pyrallis.wrap()
def main(cfg: RunConfig):
    """
    Main function. Calls the function to transfer style to either a dataset of images or a single image.
    """
    random.seed(0)
    cfg.output_path = cfg.output_path / cfg.name / f"ts{cfg.num_timesteps}_skip{cfg.skip_steps}_{'class-' if cfg.adain_class else ''}adain"
    cfg.output_path.mkdir(parents=True, exist_ok=True)
    
    # Transfer style to a dataset
    transfer_dataset(cfg=cfg, nb_img_per_style=cfg.nb_img_per_style, sampling='ordered-repeat')
    
    # Uncomment to transfer style to a single image
    # transfer_one_image(cfg=cfg, content_img_number=2605, style_name="GOPR0122_frame_000234")


def transfer_dataset(cfg: RunConfig, nb_img_per_style=5, sampling='ordered-repeat'):
    """
    Transfers style to a dataset of content images. Selects images based on a sampling strategy.
    Args:
        cfg: Model configuration.
        nb_img_per_style: Number of content images to transfer style to for each style image.
        sampling: Sampling strategy ('rcs', 'ordered', 'ordered-repeat' or 'uniform').
    """
    style_image_root = cfg.style_image_path / "images"
    style_label_root = cfg.style_image_path / "labels"

    set_seed(cfg.seed)
    model = CACTIFModel(cfg)
    model.pipe.scheduler.set_timesteps(cfg.num_timesteps)
    model.config = cfg

    # Load style labels
    list_style_labels = os.listdir(style_label_root)
    list_style_labels, _ = open_style_labels(list_style_labels, style_label_root, palettes.CITYSCAPES)
    total_images = nb_img_per_style * len(list_style_labels)

    if sampling == 'rcs':
        # Rare class sampling
        rcs_class_temp = 0.01
        rcs_min_pixels = 3000
        rcs_classes, rcs_classprob = get_rcs_class_probs(cfg.content_image_path, rcs_class_temp)

        with open(cfg.content_image_path / 'samples_with_class.json', 'r') as of:
            samples_with_class_and_n = json.load(of)
        
        # Filter samples with sufficient pixels
        samples_with_class_and_n = {
            int(k): v for k, v in samples_with_class_and_n.items() if int(k) in rcs_classes
        }
        samples_with_class = {c: [f for f, p in samples_with_class_and_n[c] if p > rcs_min_pixels] for c in rcs_classes}

        for c in samples_with_class:
            assert len(samples_with_class[c]) > 0

        # Sample images based on class probabilities
        random_class = np.random.choice(rcs_classes, size=total_images, replace=True, p=rcs_classprob)
        samples_content = [np.random.choice(samples_with_class[c], replace=False) for c in random_class]
    elif sampling == 'ordered':
        # Ordered sampling with unique images for each style
        samples_content = list(range(cfg.img_range.start, cfg.img_range.start + total_images))
    elif sampling == 'ordered-repeat':
        # Ordered sampling with the same images for each style
        samples_content = list(range(cfg.img_range.start, cfg.img_range.start + nb_img_per_style)) * len(list_style_labels)
    else:
        # Uniform random sampling
        samples_content = np.random.choice(cfg.img_range.end, total_images, replace=False)

    it = iter(samples_content)
    output_data = cfg.output_path

    for img_style in tqdm(list_style_labels):
        cfg.output_path = output_data / img_style.split("_gt")[0]
        cfg.output_path.mkdir(parents=True, exist_ok=True)

        for i in tqdm(range(nb_img_per_style)):
            img_number = next(it)
            if sampling == 'rcs':
                img_number = int(img_number.split("_label")[0].split("/")[-1])
            
            img_name = f"{img_number:0>5d}.png"
            path_content_img = cfg.content_image_path / "images" / img_name
            path_content_label = cfg.content_image_path / "labels" / img_name

            # Handle different style naming conventions
            name_close_style = img_style.split("_gt")[0]
            path_style_img = style_image_root / f'{name_close_style}_leftImg8bit.png'
            if path_style_img.exists():
                path_style_label = style_label_root / f'{name_close_style}_gtFine_color.png'
            else:
                path_style_img = style_image_root / f'{name_close_style}_rgb_anon.png'
                path_style_label = style_label_root / f'{name_close_style}_gt_labelColor.png'

            # Perform style transfer
            load_latents_couple(cfg, model, path_content_img, path_content_label, path_style_img, path_style_label, img_number)

    # Save configuration
    pyrallis.dump(cfg, open(cfg.output_path / 'config.yaml', 'w'))


def transfer_one_image(cfg: RunConfig, content_img_number=3, style_name="zurich_000087_000019"):
    """
    Transfer style to a single image.
    """
    style_image_root = cfg.style_image_path / "images/"
    style_label_root = cfg.style_image_path / "labels/"

    set_seed(cfg.seed)
    model = CACTIFModel(cfg)
    model.pipe.scheduler.set_timesteps(cfg.num_timesteps)
    model.config = cfg
    model.filter_perc = cfg.filter_perc

    img_name = f"{content_img_number:0>5d}.png"
    path_content_img = cfg.content_image_path / "images" / img_name
    path_content_label = cfg.content_image_path / "labels" / img_name

    # Relabel content image
    content_label = Image.open(path_content_label)
    content_label = relabel_image(content_label, palettes.GTA)

    # Handle different style naming conventions
    path_style_img = style_image_root / f'{style_name}_leftImg8bit.png'
    if path_style_img.exists():
        path_style_label = style_label_root / f'{style_name}_gtFine_color.png'
    else:
        path_style_img = style_image_root / f'{style_name}_rgb_anon.png'
        path_style_label = style_label_root / f'{style_name}_gt_labelColor.png'

    cfg.output_path = cfg.output_path / style_name
    # Perform style transfer
    load_latents_couple(cfg, model, path_content_img, path_content_label, path_style_img, path_style_label, content_img_number)


def load_latents_couple(cfg: RunConfig, model: CACTIFModel, path_content_img, path_content_label, path_style_img, path_style_label, img_number=3):
    """
    Loads and processes latents for both content and style images.
    """
    with torch.no_grad():
        # Process labels for style and content
        label_style, label_style_adain = process_label(path_style_label, palettes.CITYSCAPES)
        model.label_style.append(label_style)
        model.label_style_adain = label_style_adain
        model.label_content, model.label_content_adain = process_label(path_content_label, palettes.GTA)

        # Load or invert images to latents
        cfg.update_latents_path(path_content_img.stem, path_style_img.stem)
        latents_style, noise_style = load_or_invert_one_image(model.pipe, cfg, img_path=path_style_img, type_img="style")
        latents_content, noise_content = load_or_invert_one_image(model.pipe, cfg, img_path=path_content_img, type_img="content")
        model.set_latents(latents_style, latents_content)
        model.set_noise(noise_style, noise_content)
        model.set_onehot_masks()

        # Run the style transfer process
        run_style_transfer(model=model, cfg=cfg, img_number=img_number)


def run_style_transfer(model: CACTIFModel, cfg: RunConfig, img_number: str):
    """
    Setups and runs the diffusion process.
    """
    init_latents, init_zs = get_init_latents_and_noises(model=model, cfg=cfg)
    model.pipe.scheduler.set_timesteps(cfg.num_timesteps)
    model.enable_edit = True  # Enable cross-image attention layers

    start_step = min(cfg.cross_attn_32_range.start, cfg.cross_attn_64_range.start)
    end_step = max(cfg.cross_attn_32_range.end, cfg.cross_attn_64_range.end)

    # Run diffusion process
    images = model.pipe(
        prompt=[cfg.prompt] * 3,
        latents=init_latents,
        guidance_scale=1.0,
        num_inference_steps=cfg.num_timesteps,
        swap_guidance_scale=cfg.swap_guidance_scale,
        callback=model.get_adain_callback(),
        eta=1,
        zs=init_zs,
        generator=torch.Generator('cuda').manual_seed(cfg.seed),
        cross_image_attention_range=Range(start=start_step, end=end_step)
    ).images

    # Save outputs
    (cfg.output_path / "transfer").mkdir(parents=True, exist_ok=True)
    (cfg.output_path / "input").mkdir(parents=True, exist_ok=True)
    (cfg.output_path / "joined").mkdir(parents=True, exist_ok=True)

    images[0].save(cfg.output_path / "transfer" / f"{cfg.name}_transfer_{img_number}.png")
    images[1].save(cfg.output_path / "input" / f"{cfg.name}_style_{img_number}.png")
    images[2].save(cfg.output_path / "input" / f"{cfg.name}_content_{img_number}.png")

    joined_images = np.concatenate(images[::-1], axis=1)
    Image.fromarray(joined_images).save(cfg.output_path / "joined" / f"{cfg.name}_joined_{img_number}.png")
    return images


# Entry point
if __name__ == '__main__':
    main()
