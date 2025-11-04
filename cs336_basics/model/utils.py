import torch
import torch.nn as nn
from collections.abc import Iterable


def SiLU(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    x = x - x.max(dim=dim, keepdim=True).values
    log_sum_exp = torch.log(torch.sum(torch.exp(x), dim=dim, keepdim=True))
    return torch.exp(x - log_sum_exp)


def cross_entropy_loss(inputs: torch.Tensor, targets: torch.Tensor) -> float:
    log_probs = -nn.functional.log_softmax(inputs, dim=-1)
    nll_loss = log_probs.gather(dim=-1, index=targets.unsqueeze(1)).mean()
    return nll_loss


def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float, eps: float = 1e-6) -> None:
    grad_norm_sq = sum(
        torch.linalg.norm(parameter.grad, ord=2) ** 2 for parameter in parameters if parameter.grad is not None
    )
    grad_norm = torch.sqrt(grad_norm_sq)
    if grad_norm > max_l2_norm:
        scaling_factor = max_l2_norm / (grad_norm + eps)
        for parameter in parameters:
            if parameter.grad is not None:
                parameter.grad.mul_(scaling_factor)
