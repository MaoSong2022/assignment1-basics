import torch
import torch.nn as nn
from einops import einsum, rearrange


def SiLU(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


class Linear(nn.Module):
    def __init__(
        self, in_features: int, out_features: int, device: torch.device | None = None, dtype: torch.dtype | None = None
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(out_features, in_features)).to(device=device, dtype=dtype)

        std = 2 / (in_features + out_features) ** 0.5
        torch.nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3 * std, b=3 * std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")


class Embedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.W = nn.Parameter(torch.zeros(num_embeddings, embedding_dim)).to(device=device, dtype=dtype)
        torch.nn.init.trunc_normal_(self.W, std=1, a=-3, b=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.W[x]


class RMSNorm(nn.Module):
    def __init__(
        self, d_model: int, eps: float = 1e-5, device: torch.device | None = None, dtype: torch.dtype | None = None
    ) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model)).to(device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight * x / (x.pow(2).mean(dim=-1, keepdim=True) + self.eps).sqrt()


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, device: torch.device | None = None, dtype: torch.dtype | None = None) -> None:
        super().__init__()
        d_ff = int(8 / 3 * d_model)
        self.W2 = nn.Parameter(torch.zeros(d_model, d_ff)).to(device=device, dtype=dtype)
        self.W3 = nn.Parameter(torch.zeros(d_ff, d_model)).to(device=device, dtype=dtype)
        self.W1 = nn.Parameter(torch.zeros(d_model, d_ff)).to(device=device, dtype=dtype)
        std = 2 / (d_model + d_ff) ** 0.5
        torch.nn.init.trunc_normal_(self.W2, mean=0.0, std=std)
        torch.nn.init.trunc_normal_(self.W3, mean=0.0, std=std)
        torch.nn.init.trunc_normal_(self.W1, mean=0.0, std=std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gated = einsum(x, self.W1, "... d_model, d_ff d_model -> ... d_ff")
        gated = SiLU(gated)
        out = einsum(x, self.W3, "... d_model, d_ff d_model -> ... d_ff")
        out = einsum(out, gated, "... d_ff, ... d_ff -> ... d_ff")
        out = einsum(out, self.W2, "... d_ff, d_model d_ff -> ... d_model")
        return out


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_pos_emb(
    in_query_or_key: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, unsqueeze_dim: int | None = None
) -> torch.Tensor:
    if unsqueeze_dim is not None:
        # (batch_size, seq_len, d_k) -> (batch_size, 1, seq_len, d_k)
        cos = cos.unsqueeze(unsqueeze_dim)
        sin = sin.unsqueeze(unsqueeze_dim)

    return in_query_or_key * cos + rotate_half(in_query_or_key) * sin


class RoPE(nn.Module):
    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        inv_freq = 1.0 / self.theta ** (torch.arange(0, d_k, 2, device=device, dtype=dtype) / d_k)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        inv_freq = self.inv_freq.float()  # (d_k_half)
        token_positions_expanded = token_positions.float()  # (seq_len)

        freqs = einsum(inv_freq, token_positions_expanded, "d_k_half, ... seq_len -> ... d_k_half seq_len")
        freqs = freqs.transpose(-1, -2)  # (seq_len, d_k_half)
        emb = torch.cat((freqs, freqs), dim=-1)  # [seq_len, d_k]
        cos = emb.cos().to(x.device, x.dtype)  # (seq_len, d_k)
        sin = emb.sin().to(x.device, x.dtype)  # (seq_len, d_k)

        return apply_rotary_pos_emb(x, cos, sin)


def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    d_k = Q.shape[-1]
    scores = einsum(Q, K, "b ... s_q d_k, b ... s_k d_k -> b ... s_q s_k")
    scores = scores / d_k ** 0.5
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))
    scores = scores.softmax(dim=-1)
    return einsum(scores, V, "b ... s_q s_k, b ... s_k d_v -> b ... s_q d_v")
