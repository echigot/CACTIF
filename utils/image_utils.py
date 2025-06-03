from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image

from config import RunConfig


def open_style_labels(list_path, style_label_root, palette):
    """
    Opens and filters style label images based on naming convention.
    
    Args:
        list_path: List of filenames.
        style_label_root: Directory containing the style label images.
        palette: Not used in current implementation but likely for future processing.

    Returns:
        A tuple of:
            - list of filenames that match criteria
            - list of opened PIL images
    """
    list_labels = []
    list_path_clean = []

    for img_style in list_path:
        if 'olor.png' in img_style.split('_')[-1]:  # Select specific labeled images
            img = Image.open(Path(style_label_root / img_style))
            list_labels.append(img)
            list_path_clean.append(img_style)

    return list_path_clean, list_labels


def load_images(cfg: RunConfig, save_path: Optional[Path] = None) -> Tuple[Image.Image, Image.Image]:
    """
    Loads and resizes style and content images.

    Args:
        cfg: Model configuration.
        save_path: Optional directory to save loaded images for inspection.

    Returns:
        A tuple of:
            - style image (resized)
            - content image (resized)
    """
    style_image = load_size(cfg.style_image_path)
    content_image = load_size(cfg.content_image_path)

    if save_path is not None:
        Image.fromarray(style_image).save(save_path / "in_style.png")
        Image.fromarray(content_image).save(save_path / "in_content.png")

    return style_image, content_image


def load_size(image_path, size: int = 512):
    """
    Loads and resizes an image while preserving a 2:1 aspect ratio (width:height).

    Args:
        image_path: File path to the image or a PIL Image object.
        size: Target height (width will be 2x height to maintain 2:1 aspect).

    Returns:
        Resized image as a NumPy array (if provided Path) or a PIL Image (if provided PIL Image).
    """
    input_is_image = False

    # Load image if path is provided
    if isinstance(image_path, (str, Path)):
        image = Image.open(str(image_path)).convert('RGB')
        width, height = image.size
        image = np.array(image)
    else:
        image = image_path
        width, height = image.size
        input_is_image = True

    # Maintain 2:1 width:height aspect ratio
    aspect = width / float(height)
    ideal_width = size * 2
    ideal_height = size
    ideal_aspect = ideal_width / float(ideal_height)

    if aspect > ideal_aspect:
        # Crop left and right to match desired aspect
        new_width = int(ideal_aspect * height)
        offset = (width - new_width) / 2
        crop_box = (offset, 0, width - offset, height)
    else:
        # Crop top and bottom
        new_height = int(width / ideal_aspect)
        offset = (height - new_height) / 2
        crop_box = (0, offset, width, height - offset)

    # Convert array back to image if necessary
    if not input_is_image:
        image = Image.fromarray(image)

    image = image.crop(crop_box)
    image = image.resize((ideal_width, ideal_height))

    # Return image as array again if it was originally loaded from path
    if not input_is_image:
        image = np.array(image)

    return image
