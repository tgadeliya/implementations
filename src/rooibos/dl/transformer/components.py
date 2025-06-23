from math import sqrt
from collections import OrderedDict

import einops
import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import Parameter


class Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype

        self.W = torch.nn.Parameter(
            torch.empty(out_features, in_features, device=device, dtype=dtype)
        )
        std_val = sqrt(2 / (in_features + out_features))
        nn.init.trunc_normal_(
            self.W, mean=0, std=std_val, a=-3 * std_val, b=3 * std_val
        )

    def forward(self, x: Tensor) -> Tensor:
        return einops.einsum(x, self.W, "... d_in, d_out d_in -> ... d_out")


class Embedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.device = device
        self.dtype = dtype

        self.E = torch.nn.Parameter(
            torch.empty(
                num_embeddings, embedding_dim, device=device, dtype=dtype
            )
        )
        nn.init.trunc_normal_(self.E, mean=0, std=1, a=-3, b=3)

    def forward(self, token_ids: Tensor) -> Tensor:
        return self.E[token_ids, :]


class RMSNorm(nn.Module):
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.d_model = d_model
        self.device = device
        self.dtype = dtype

        self.g = torch.nn.Parameter(
            torch.ones(d_model, device=device, dtype=dtype)
        )

    def forward(self, x: Tensor) -> Tensor:
        x_dtype = x.dtype
        x = x.to(torch.float32)

        x_rms = (
            x.square().sum(dim=-1, keepdim=True) / self.d_model + self.eps
        ).sqrt()
        result = einops.einsum(
            (x / x_rms), self.g, " ... d_model, d_model -> ... d_model"
        )
        return result.to(x_dtype)


class SiLU(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return x * torch.sigmoid(x)


class SwiGLU(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.d_ff = d_ff  # int(((d_model * 8 / 3) + 63) // 64) * 64
        self.d_model = d_model
        self.device = device if device else torch.device("cpu")
        self.dtype = dtype

        self.W1 = torch.nn.Parameter(
            torch.empty(self.d_ff, self.d_model, device=device, dtype=dtype)
        ).to(self.device)
        self.W2 = torch.nn.Parameter(
            torch.empty(self.d_model, self.d_ff, device=device, dtype=dtype)
        ).to(self.device)
        self.W3 = torch.nn.Parameter(
            torch.empty(self.d_ff, self.d_model, device=device, dtype=dtype)
        ).to(self.device)
        self.act_func = SiLU()

        std_val = sqrt(2 / (self.d_model + self.d_ff))
        nn.init.trunc_normal_(
            self.W1, mean=0, std=std_val, a=-3 * std_val, b=3 * std_val
        )
        nn.init.trunc_normal_(
            self.W2, mean=0, std=std_val, a=-3 * std_val, b=3 * std_val
        )
        nn.init.trunc_normal_(
            self.W3, mean=0, std=std_val, a=-3 * std_val, b=3 * std_val
        )

    def forward(self, x: Tensor) -> Tensor:
        out = einops.einsum(x, self.W1, "... d_model, d_ff d_model -> ... d_ff")
        out1 = self.act_func(out)
        out2 = einops.einsum(
            x, self.W3, "... d_model, d_ff d_model -> ... d_ff"
        )
        out = out1 * out2
        out = einops.einsum(
            out, self.W2, "... d_ff, d_model d_ff -> ... d_model"
        )
        return out


class RotaryPositionalEmbedding(nn.Module):
    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device
        R = self._generate_rotary_matrix(
            torch.ones(max_seq_len, d_k, d_k, dtype=torch.float32)
        )
        self.register_buffer("R", R, persistent=False)

    def forward(self, x: Tensor, token_positions: Tensor) -> Tensor:
        return einops.einsum(
            x,
            self.R[token_positions, ...],
            "... seq_len d_k, seq_len d_kk d_k -> ... seq_len d_kk",
        )

    def _generate_rotary_matrix(self, rm):
        for i in range(self.max_seq_len):
            rm[i, :, :] = self.gen_pos_i(i)
        return rm

    def gen_pos_i(self, i):
        pows = (2 * torch.arange(self.d_k // 2, dtype=torch.float32)) / self.d_k
        thetai = i / (self.theta**pows)
        ts = torch.sin(thetai)
        tc = torch.cos(thetai)
        rot_matrices = torch.empty(
            (self.d_k // 2, 2, 2), device=self.device, dtype=torch.float32
        )
        rot_matrices[:, 0, 0] = tc
        rot_matrices[:, 0, 1] = -ts
        rot_matrices[:, 1, 0] = ts
        rot_matrices[:, 1, 1] = tc
        return torch.block_diag(*rot_matrices).to(self.device)


def softmax(x: Tensor, dim: int = -1) -> Tensor:
    xs = torch.exp(x - x.max())
    xs /= xs.sum(dim=-1, keepdim=True)
    return xs


def scaled_dot_product_attention(
    Q: Tensor, K: Tensor, V: Tensor, mask: Tensor | None = None
) -> Tensor:
    KQ = einops.einsum(
        K, Q, "... seq_lenk d , ... seq_lenq d -> ... seq_lenq seq_lenk"
    ) / (Q.size()[-1] ** 0.5)
    if mask is not None:
        KQ.masked_fill_(~mask, -torch.inf)
    KQ = softmax(KQ, dim=-1)
    return einops.einsum(
        KQ, V, "... seq_lenq seq_lenk , ... seq_lenk d_v -> ... seq_lenq d_v"
    )


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        pos_emb_func: RotaryPositionalEmbedding | None = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads

        d_k = d_v = d_model // num_heads
        self.q_proj = Parameter(torch.empty(num_heads * d_k, d_model))
        self.k_proj = Parameter(torch.empty(num_heads * d_k, d_model))
        self.v_proj = Parameter(torch.empty(num_heads * d_v, d_model))
        self.o_proj = Parameter(torch.empty(d_model, num_heads * d_v))
        self.pos_emb_func = pos_emb_func

        std_val = sqrt(2 / (self.d_model + num_heads * d_k))
        nn.init.trunc_normal_(
            self.q_proj, mean=0, std=std_val, a=-3 * std_val, b=3 * std_val
        )
        nn.init.trunc_normal_(
            self.k_proj, mean=0, std=std_val, a=-3 * std_val, b=3 * std_val
        )
        nn.init.trunc_normal_(
            self.v_proj, mean=0, std=std_val, a=-3 * std_val, b=3 * std_val
        )
        nn.init.trunc_normal_(
            self.o_proj, mean=0, std=std_val, a=-3 * std_val, b=3 * std_val
        )

    def forward(
        self, x: Tensor, token_positions: Tensor | None = None
    ) -> Tensor:
        q = einops.rearrange(
            einops.einsum(x, self.q_proj, "... d_in, d_out d_in  -> ... d_out"),
            "... seq_len (nh d_k) -> ... nh seq_len d_k",
            nh=self.num_heads,
        )

        k = einops.rearrange(
            einops.einsum(x, self.k_proj, "... d_in, d_out d_in  -> ... d_out"),
            "... seq_len (nh d_k) -> ... nh seq_len d_k",
            nh=self.num_heads,
        )

        v = einops.rearrange(
            einops.einsum(x, self.v_proj, "... d_in, d_out d_in  -> ... d_out"),
            "... seq_len (nh d_k) -> ... nh seq_len d_k",
            nh=self.num_heads,
        )

        seq_len_q, seq_len_k = q.size()[-2], k.size()[-2]
        if self.pos_emb_func:
            if token_positions is None:
                token_positions = torch.arange(seq_len_q).reshape(1, -1)
            q = self.pos_emb_func(q, token_positions=token_positions[0])
            k = self.pos_emb_func(k, token_positions=token_positions[0])

        mask = ~torch.triu(
            input=torch.ones(seq_len_q, seq_len_k, device=q.device), diagonal=1
        ).bool()

        attn = scaled_dot_product_attention(q, k, v, mask=mask)
        return einops.einsum(
            einops.rearrange(
                attn, "... nh seq_len d_k -> ... seq_len (nh d_k)"
            ),
            self.o_proj,
            "bs sl d, d_model d  -> bs sl d_model",
        )


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        theta: float,
        max_seq_len: int,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.d_k = d_model // num_heads
        pos_emb_func = RotaryPositionalEmbedding(
            theta=theta, max_seq_len=max_seq_len, d_k=self.d_k
        )
        self.attn = MultiHeadAttention(
            d_model=d_model, num_heads=num_heads, pos_emb_func=pos_emb_func
        )
        self.ffn = SwiGLU(d_model=d_model, d_ff=d_ff)
        self.ln1 = RMSNorm(d_model=d_model)
        self.ln2 = RMSNorm(d_model=d_model)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x
