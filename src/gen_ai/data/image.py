from torch import Tensor
from matplotlib.axes import Axes

ImageShape = tuple[int, int]

def show_image(image: Tensor, ax: Axes):
    image = (image.detach().cpu().permute(1, 2, 0) + 1) / 2
    ax.imshow(image)
    ax.axis("off")