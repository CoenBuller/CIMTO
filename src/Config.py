from dataclasses import dataclass

@dataclass 
class phantomConfig:
    max_gray: int = 255
    min_gray: int = 0
    img_shape: tuple[int, int] = (512, 512)
    seed: int = 69
    save_dir: str = "Phantoms/"