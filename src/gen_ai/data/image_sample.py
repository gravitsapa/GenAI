from typing import NamedTuple

from torch import Tensor
from matplotlib.axes import Axes

class ImageSample(NamedTuple):
    image: Tensor
    tags: str

def show_sample(image_sample: ImageSample, ax: Axes):
    image = (image_sample.image.detach().cpu().permute(1, 2, 0) + 1) / 2
    ax.imshow(image)
    ax.axis("off")

