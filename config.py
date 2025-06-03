from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple, Optional


class Range(NamedTuple):
    start: int
    end: int

@dataclass
class RunConfig:
    # Style image path
    style_image_path: Path = Path("./data/style/")
    # Content image path
    content_image_path: Path = Path("./data/gta/")
    # Output path
    output_path: Path = Path('./output')
    # Path to save the inverted latent codes
    latents_path: Path = Path("./latents")
    
    # Random seed
    seed: int = 42
    # Input prompt for inversion
    prompt: Optional[str] = ""
    # Number of timesteps
    num_timesteps: int = 50
    # Number of steps to skip in the denoising process
    skip_steps: int = 30
    # Whether to load previously saved inverted latent codes
    load_latents: bool = True
    
    # Timesteps to apply cross-attention on 64x128 layers
    cross_attn_64_range: Range = Range(start=1, end=50)
    # Timesteps to apply cross-attention on 32x64 layers
    cross_attn_32_range: Range = Range(start=1, end=50) 
    # Swap guidance scale
    swap_guidance_scale: float = 1
    # Attention contrasting strength
    contrast_strength: float = 1.67

    # Style transfer name: "CACTIF" "CACTI" "cross_image"
    name: str = "CACTIF"
    # Apply the filtering operation of CACTIF
    filtering: bool = True
    # Percentage p of filtering in CACTIF
    filter_perc: int = 0.25
    # Apply the cross-attention operation introduced by cross-image attention [Alaluf et al. (2024)]
    cross_attention: bool = True
    # Apply AdaIN per class
    adain_class: bool = True
    # Timesteps to apply class AdaIn
    class_adain_range: Range = Range(start=5, end=15)
    
    # part of the GTA5 dataset to consider
    img_range: Range = Range(start=1, end=24966)
    # Number of images generated for each style image
    nb_img_per_style: int = 5 # 250 for domain adaptation tasks
    
    def __post_init__(self):
        self.latents_path = self.latents_path / f"latents_{self.num_timesteps}"
        Path(self.latents_path / "style").mkdir(parents=True, exist_ok=True)
        Path(self.latents_path / "content").mkdir(parents=True, exist_ok=True)

    def update_latents_path(self, name_content_image, name_style_image):
        # Define the paths to store the inverted latents to
        self.style_latent_save_path = self.latents_path / "style" / f"{name_style_image}.pt"
        self.content_latent_save_path = self.latents_path / "content" / f"{name_content_image}.pt"
        