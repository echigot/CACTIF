import os
import json

from PIL import Image
import numpy as np
import torch

from .image_utils import load_size


def relabel_image(image: Image, palette: list):
    """
    Converts an RGB image into a label map using a given dataset palette.
    
    Args:
        image (PIL.Image): The input image.
        palette (list): List of values representing label classes for a specific dataset.

    Returns:
        np.ndarray: Image array with each pixel mapped to its palette index, or 255 if not found.
    """
    new_pixels = []
    
    for pixel in image.getdata():
        # Extract RGB values depending on image mode
        if image.mode == "P":
            rgb_pix = pixel
        elif image.mode == "RGBA":
            rgb_pix = list(pixel[:-1])
        else:  # "RGB"
            rgb_pix = list(pixel)

        # Assign the class index from the palette or 255 for unknown classes
        if rgb_pix in palette:
            new_pixels.append(palette.index(rgb_pix))
        else:
            new_pixels.append(255)
    
    # Create a new image from the reclassified pixel data
    new_image = Image.new(mode="P", size=image.size)
    new_image.putdata(new_pixels)
    
    return np.array(new_image)


def process_label(path, palette):
    """
    Loads and processes a label image by resizing and relabeling it.

    Args:
        path (str or Path): File path to the label image.
        palette (list): List of values representing label classes for a specific dataset.

    Returns:
        tuple: (main label array, resized label array for AdaIN)
    """
    label = Image.open(path)
    label = load_size(label, size=512)  # Resize main label image
    label_adain = label.resize((128, 64))  # Resize version for AdaIN module

    # Relabel using the provided palette
    label = relabel_image(label, palette)
    label_adain = relabel_image(label_adain, palette)

    return label, label_adain


def get_rcs_class_probs(data_root, temperature):
    """
    Computes class sampling probabilities using Rare Class Sampling (RCS), from the DAFormer repository [Hoyer et al. (2022)].
    
    Args:
        data_root (str or Path): Root directory containing `sample_class_stats.json`.
        temperature (float): Temperature parameter for softmax to control distribution sharpness.

    Returns:
        tuple: (list of class indices, class sampling probabilities as numpy array)
    """
    with open(os.path.join(data_root, 'sample_class_stats.json'), 'r') as of:
        sample_class_stats = json.load(of)

    overall_class_stats = {}
    
    for s in sample_class_stats:
        s.pop('file')
        for c, n in s.items():
            c = int(c)
            overall_class_stats[c] = overall_class_stats.get(c, 0) + n

    overall_class_stats = {
        k: v for k, v in sorted(overall_class_stats.items(), key=lambda item: item[1])
    }

    freq = torch.tensor(list(overall_class_stats.values()), dtype=torch.float32)
    freq = freq / torch.sum(freq)  
    freq = 1 - freq  
    freq = torch.softmax(freq / temperature, dim=-1)

    return list(overall_class_stats.keys()), freq.numpy()
