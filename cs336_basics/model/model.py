from einops import einsum, rearrange
import math
import torch
import torch.nn as nn

from cs336_basics.model import utils


class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))

        # truncate normalization
        std = math.sqrt(2.0 / (in_features + out_features))
        nn.init.trunc_normal_(self.weight, mean=0, std=std, a=-3 * std, b=3 * std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")


class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim))

        # truncate normalization
        nn.init.trunc_normal_(self.weight, mean=0, std=1, a=-3, b=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight[x]


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 13 - 5):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_dtype = x.dtype
        x.dtype = torch.float32
        variance = torch.pow(2).mean(-1, keepdim=True)
        output = x * torch.rsqrt(variance + self.eps) * self.scale
        return output.to(x_dtype)


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.W1 = Linear(d_model, d_ff)
        self.W3 = Linear(d_model, d_ff)
        self.W2 = Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.W2(utils.SiLU(self.W1(x)) * self.W3(x))


class RotaryPositionEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int):
        super().__init__()

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        pass


def apply_rotary_pos_emb(q, k, cos, sin, position_ids) -> tuple[torch.Tensor, torch.Tensor]:
    pass


def scaled_dot_product_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor | None
) -> torch.Tensor:
    attention_score = einsum(q, k, "... queries d_k, ... keys d_k -> ... queries keys")
    if mask is not None:
        attention_score.masked_fill_(mask == 0, float("-inf"))

    attention_weight = utils.softmax(attention_score, dim=-1)

    return einsum(attention_weight, v, "... queries keys, ... keys d_v -> ... queries d_v")


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass


class DecodeLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int):
        super().__init__()
        self.pre_norm = RMSNorm(d_model)
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.post_norm = RMSNorm(d_model)
        self.FFN = SwiGLU(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.pre_norm(x)
        x = self.attention(x)
        x = x + residual

        residual = x
        x = self.post_norm(x)
        x = self.FFN(x)
        x = x + residual

        return x


class Model(nn.Module):
    def __init__(self, vocab_size: int, context_length: int, num_layers: int, d_model: int, num_heads: int, d_ff: int):
        super().__init__()

        self.embedding = Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([DecodeLayer(d_model, num_heads, d_ff) for _ in range(num_layers)])
        self.norm = RMSNorm(d_model)
        self.lm_head = Linear(d_model, vocab_size)

    def forward(self, token_ids) -> torch.Tensor:
        hidden_states = self.embedding(token_ids)

        for decode_layer in self.layerss:
            hidden_states = decode_layer(hidden_states)

        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)

        probs = utils.softmax(logits)

        return probs
