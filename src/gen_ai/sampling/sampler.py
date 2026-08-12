from typing import Optional

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

import torch
import torch.nn as nn


from gen_ai.data.image_sample import show_image
 
from gen_ai.models.generative import ImageGenerativeModel
from gen_ai.exceptions import require

class Sampler:
    def __init__(
        self,
        model: ImageGenerativeModel,
    ):
        self.model = model


    @torch.inference_mode()
    def sample_grid(
        self,
        nrows: int=5, 
        ncols: int=5,
        size_per_image: int=2,
    ) -> Figure:
        require(nrows > 0, ValueError, f"nrows must be positive, got {nrows}")
        require(ncols > 0, ValueError, f"ncols must be positive, got {ncols}")
        require(
            size_per_image > 0,
            ValueError,
            f"size_per_image must be positive, got {size_per_image}",
        )
        fig, axs = plt.subplots(
            nrows, 
            ncols, 
            figsize=(nrows*size_per_image, ncols*size_per_image)
        )

        batch_size = nrows * ncols
        self.model.eval()
        samples = self.model.sample(batch_size)

        for row in range(nrows):
            for col in range(ncols):
                index = row * ncols + col
                show_image(samples[index], axs[row][col])

        return fig

