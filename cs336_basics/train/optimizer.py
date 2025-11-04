import math
import torch


class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr: float, betas: tuple[float, float], weight_decay: float, eps: float):
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay, eps=eps)
        super().__init__(params=params, defaults=defaults)

    def step(self):
        for group in self.param_groups:
            # 获取当前组的超参数（每个组可能不同）
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            weight_decay = group["weight_decay"]
            eps = group["eps"]
            for p in group["params"]:
                if p.grad is None:
                    continue  # 跳过无梯度的参数

                grad = p.grad.data

                if p not in self.state:
                    self.state[p] = {
                        "first_moment": torch.zeros_like(p.data),
                        "second_moment": torch.zeros_like(p.data),
                        "step": 0,
                    }

                first_moment = self.state[p]["first_moment"]
                second_moment = self.state[p]["second_moment"]
                step = self.state[p]["step"]

                step += 1
                first_moment = beta1 * first_moment + (1 - beta1) * grad
                second_moment = beta2 * second_moment + (1 - beta2) * grad**2

                corrected_lr = lr * math.sqrt(1 - beta2**step) / (1 - beta1**step)
                p.data = p.data - corrected_lr * first_moment / (torch.sqrt(second_moment) + eps)
                p.data *= 1 - lr * weight_decay

                self.state[p] = {
                    "first_moment": first_moment,
                    "second_moment": second_moment,
                    "step": step,
                }


def get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    if it < warmup_iters:
        return max_learning_rate * it / warmup_iters
    elif it < cosine_cycle_iters:
        return min_learning_rate + 0.5 * (
            1 + math.cos((it - warmup_iters) / (cosine_cycle_iters - warmup_iters) * torch.pi)
        ) * (max_learning_rate - min_learning_rate)
    else:
        return min_learning_rate
