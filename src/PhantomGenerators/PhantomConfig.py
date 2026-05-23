from dataclasses import dataclass

@dataclass 
def config:
    max_gray = 255
    min_gray = 0
    img_shape = (512, 512)
    seed = 69