from typing import NamedTuple

from torch import Tensor
from matplotlib.axes import Axes

class ImageSample(NamedTuple):
    image: Tensor
    tags: str

def show_sample(image_sample: ImageSample, ax: Axes):
    ax.imshow(image_sample.image.cpu().permute(1, 2, 0))
    ax.axis("off")

