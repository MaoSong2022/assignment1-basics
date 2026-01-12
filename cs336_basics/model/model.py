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
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_dtype = x.dtype
        x = x.to(torch.float32)
        variance = torch.pow(x, 2).mean(-1, keepdim=True)
        output = x * torch.rsqrt(variance + self.eps) * self.weight
        return output.to(x_dtype)


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w1 = Linear(d_model, d_ff)
        self.w3 = Linear(d_model, d_ff)
        self.w2 = Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(utils.SiLU(self.w1(x)) * self.w3(x))


class RotaryPositionEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int):
        super().__init__()
        self.base = theta
        self.dim = d_k
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        self.max_seq_len = max_seq_len

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        t = token_positions.float()
        freqs = einsum(t, self.inv_freq, "... i, j -> ... i j")
        return freqs.cos(), freqs.sin()


def apply_rotary_pos_emb(x, cos, sin) -> torch.Tensor:
    x_even = x[..., ::2]  # (seq_len, d_k//2)
    x_odd = x[..., 1::2]  # (seq_len, d_k//2)
    odds = cos * x_even - sin * x_odd
    evens = sin * x_even + cos * x_odd
    stacked = torch.stack((odds, evens), dim=-2)  # (seq_len, d_k//2, 2)
    stacked_trans = rearrange(stacked, "... double d_k_half -> ... d_k_half double")  # (seq_len, 2, d_k//2)
    out = rearrange(stacked_trans, "... d_k_half double -> ... (d_k_half double)")
    return out


def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None,
    cos: torch.Tensor = None,
    sin: torch.Tensor = None,
) -> torch.Tensor:
    if cos is not None and sin is not None:
        q = apply_rotary_pos_emb(q, cos, sin)
        k = apply_rotary_pos_emb(k, cos, sin)

    attention_score = einsum(q, k, "... queries d_k, ... keys d_k -> ... queries keys")
    scale = math.sqrt(q.shape[-1])
    if mask is not None:
        attention_score.masked_fill_(mask == 0, float("-inf"))

    attention_weight = utils.softmax(attention_score / scale, dim=-1)

    return einsum(attention_weight, v, "... queries keys, ... keys d_v -> ... queries d_v")


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.q_proj = Linear(d_model, d_model)
        self.k_proj = Linear(d_model, d_model)
        self.v_proj = Linear(d_model, d_model)
        self.output_proj = Linear(d_model, d_model)

    def forward(self, x: torch.Tensor, cos: torch.Tensor = None, sin: torch.Tensor = None) -> torch.Tensor:
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        q = rearrange(
            q,
            "... seq_len (num_heads d_head) -> ... num_heads seq_len d_head",
            num_heads=self.num_heads,
            d_head=self.d_model // self.num_heads,
        )
        k = rearrange(
            k,
            "... seq_len (num_heads d_head) -> ... num_heads seq_len d_head",
            num_heads=self.num_heads,
            d_head=self.d_model // self.num_heads,
        )
        v = rearrange(
            v,
            "... seq_len (num_heads d_head) -> ... num_heads seq_len d_head",
            num_heads=self.num_heads,
            d_head=self.d_model // self.num_heads,
        )

        mask = torch.tril(torch.ones(q.shape[-2], q.shape[-2], device=q.device))

        attn_output = scaled_dot_product_attention(q, k, v, mask=mask, cos=cos, sin=sin)
        attn_output = rearrange(attn_output, "... num_heads seq_len d_head -> ... seq_len (num_heads d_head)")
        return self.output_proj(attn_output)


class DecodeLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int):
        super().__init__()
        self.ln1 = RMSNorm(d_model)
        self.attn = MultiHeadAttention(d_model, num_heads)
        self.ln2 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, d_ff)

    def forward(self, x: torch.Tensor, cos: torch.Tensor = None, sin: torch.Tensor = None) -> torch.Tensor:
        residual = x
        x = self.ln1(x)
        x = self.attn(x, cos, sin)
        x = x + residual

        residual = x
        x = self.ln2(x)
        x = self.ffn(x)
        x = x + residual

        return x


class Model(nn.Module):
    def __init__(
        self, vocab_size: int, num_layers: int, d_model: int, num_heads: int, d_ff: int, max_seq_len: int, theta: float
    ):
        super().__init__()

        self.token_embeddings = Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([DecodeLayer(d_model, num_heads, d_ff) for _ in range(num_layers)])
        self.ln_final = RMSNorm(d_model)
        self.lm_head = Linear(d_model, vocab_size)

        d_heads = d_model // num_heads
        self.rotary_embedding = RotaryPositionEmbedding(theta, d_heads, max_seq_len)

    def forward(self, token_ids) -> torch.Tensor:
        hidden_states = self.token_embeddings(token_ids)
        token_positions = torch.arange(hidden_states.shape[-2], device=hidden_states.device)
        cos, sin = self.rotary_embedding(hidden_states, token_positions)

        for decode_layer in self.layers:
            hidden_states = decode_layer(hidden_states, cos, sin)

        hidden_states = self.ln_final(hidden_states)
        logits = self.lm_head(hidden_states)

        return logits

    @torch.no_grad()
    def generate(
        self,
        token_ids: torch.Tensor,
        max_new_tokens: int,
        temperature=1.0,
        top_k=None,
        top_p=None,
        eos_token_id: int = 0,
    ) -> torch.Tensor:
        self.eval()
        batch_size = token_ids.size(0)
        device = token_ids.device
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        for _ in range(max_new_tokens):
            logits = self(token_ids)
            next_token_logits = logits[:, -1, :] / temperature

            if top_k is not None and top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = -float("Inf")

            if top_p is not None and 0.0 < top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True, dim=-1)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

                # Mask tokens where cumulative prob exceeds top_p
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift mask right to keep the first token that exceeds the threshold
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = False

                # Scatter the mask back to the original logit order
                for b in range(batch_size):
                    indices_to_remove = sorted_indices[b][sorted_indices_to_remove[b]]
                    next_token_logits[b, indices_to_remove] = -float("Inf")

            # Sample
            probs = torch.softmax(next_token_logits, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1)  # (batch, 1)

            # Stop if all sequences hit EOS (only if batch_size > 1)
            finished |= next_tokens.squeeze(-1) == eos_token_id

            token_ids = torch.cat([token_ids, next_tokens], dim=-1)

            if finished.all():
                break

        return token_ids
