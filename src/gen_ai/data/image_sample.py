from typing import NamedTuple

from matplotlib.axes import Axes
from torch import Tensor

from gen_ai.data.image import show_image

class ImageSample(NamedTuple):
    image: Tensor
    tags: str


def show_sample(image_sample: ImageSample, ax: Axes):
    show_image(image_sample.image, ax)
