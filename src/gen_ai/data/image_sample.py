from typing import NamedTuple

from torch import Tensor
from matplotlib.axes import Axes


class ImageSample(NamedTuple):
    image: Tensor
    tags: str


def show_image(image: Tensor, ax: Axes):
    image = (image.detach().cpu().permute(1, 2, 0) + 1) / 2
    ax.imshow(image)
    ax.axis("off")


def show_sample(image_sample: ImageSample, ax: Axes):
    show_image(image_sample.image, ax)
