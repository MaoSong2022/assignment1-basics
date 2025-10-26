import torch


def SiLU(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


def softmax(x: torch.Tensor) -> torch.Tensor:
    pass
