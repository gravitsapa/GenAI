
from dataclasses import dataclass, field
from torch import Tensor
from matplotlib.axes import Axes

@dataclass
class ImageSample:
    image: Tensor
    tags: str = field(default="")

def show_sample(image_sample: ImageSample, ax: Axes):
    ax.imshow(image_sample.image.permute(1, 2, 0))
    ax.axis("off")

