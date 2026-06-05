from dataclasses import dataclass
from typing import Any

@dataclass 
class phantomConfig:
    gray_values: tuple[Any, ...] = (0., 85., 170, 255.)
    max_gray: int | float = max(gray_values)
    min_gray: int  | float = min(gray_values)


    img_shape: tuple[int, int] = (512, 512)
    seed: int = 69
    save_dir: str = "TestPhantoms/"
