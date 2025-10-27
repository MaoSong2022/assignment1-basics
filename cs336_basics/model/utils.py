import torch


def SiLU(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    x = x - x.max(dim=dim, keepdim=True).values
    log_sum_exp = torch.log(torch.sum(torch.exp(x), dim=dim, keepdim=True))
    return torch.exp(x - log_sum_exp)
