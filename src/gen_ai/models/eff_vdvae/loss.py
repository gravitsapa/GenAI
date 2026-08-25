import numpy as np
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F


def diag_normal_kl_divergence(
    q_params: tuple[Tensor, Tensor],
    p_params: tuple[Tensor, Tensor],
):
    mu_q, std_q = q_params
    mu_p, std_p = p_params

    term1 = std_q / std_p
    term2 = (mu_q - mu_p) / std_p

    independent = 0.5 * (
        term1 * term1 + term2 * term2
    ) - 0.5 - torch.log(term1)

    loss = torch.sum(independent)
    return loss


def logistic_mixture_ll(
    input_tensor: Tensor, # B, H, W
    logit_probs: Tensor, # B, M, H, W
    means: Tensor, # B, M, H, W
    scales: Tensor, # B, M, H, W
):
    B, M, H, W = logit_probs.size()

    input_expanded = input_tensor.unsqueeze(1).expand(-1, M, -1, -1)
    y = (input_expanded - means) / scales

    l = logit_probs -y - torch.log(scales) - 2 * F.softplus(-y)

    return torch.logsumexp(l, dim=1)
