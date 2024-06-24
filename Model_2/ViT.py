from __future__ import annotations

from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from nni.mutable import Categorical, MutableExpression, ensure_frozen, Mutable
from nni.nas.nn.pytorch import (
    ParametrizedModule,MutableModule, MutableLayerNorm
)
from nni.nas.space import current_model
from nni.nas.profiler.pytorch.flops import FlopsResult
from nni.nas.profiler.pytorch.utils import MutableShape, ShapeTensor, profiler_leaf_module
from nni.nas.hub.pytorch.utils.nn import DropPath
from KANLinear import Mutable_KAN

class RelativePosition2D(nn.Module):
    """The implementation of relative position encoding for 2D image feature maps.

    Used in :class:`RelativePositionSelfAttention`.
    """

    def __init__(self, head_embed_dim: int, length: int = 14) -> None:
        super().__init__()
        self.head_embed_dim = head_embed_dim
        self.length = length
        self.embeddings_table_v = nn.Parameter(torch.randn(length * 2 + 2, head_embed_dim))
        self.embeddings_table_h = nn.Parameter(torch.randn(length * 2 + 2, head_embed_dim))

        nn.init.trunc_normal_(self.embeddings_table_v, std=.02)
        nn.init.trunc_normal_(self.embeddings_table_h, std=.02)

    def forward(self, length_q, length_k):
        # remove the first cls token distance computation
        length_q = length_q - 1
        length_k = length_k - 1
        # init in the device directly, rather than move to device
        range_vec_q = torch.arange(length_q, device=self.embeddings_table_v.device)
        range_vec_k = torch.arange(length_k, device=self.embeddings_table_v.device)
        # compute the row and column distance
        length_q_sqrt = int(length_q ** 0.5)
        distance_mat_v = (
            torch.div(range_vec_k[None, :], length_q_sqrt, rounding_mode='trunc') -
            torch.div(range_vec_q[:, None], length_q_sqrt, rounding_mode='trunc')
        )
        distance_mat_h = (range_vec_k[None, :] % length_q_sqrt - range_vec_q[:, None] % length_q_sqrt)
        # clip the distance to the range of [-length, length]
        distance_mat_clipped_v = torch.clamp(distance_mat_v, - self.length, self.length)
        distance_mat_clipped_h = torch.clamp(distance_mat_h, - self.length, self.length)

        # translate the distance from [1, 2 * length + 1], 0 is for the cls token
        final_mat_v = distance_mat_clipped_v + self.length + 1
        final_mat_h = distance_mat_clipped_h + self.length + 1
        # pad the 0 which represent the cls token
        final_mat_v = F.pad(final_mat_v, (1, 0, 1, 0), "constant", 0)
        final_mat_h = F.pad(final_mat_h, (1, 0, 1, 0), "constant", 0)

        final_mat_v = final_mat_v.long()
        final_mat_h = final_mat_h.long()
        # get the embeddings with the corresponding distance
        embeddings = self.embeddings_table_v[final_mat_v] + self.embeddings_table_h[final_mat_h]

        return embeddings


@profiler_leaf_module
class RelativePositionSelfAttention(MutableModule):
    """
    This class is designed to support the `relative position <https://arxiv.org/pdf/1803.02155v2.pdf>`__ in attention.

    Different from the absolute position embedding,
    the relative position embedding encodes relative distance between input tokens and learn the pairwise relations of them.
    It is commonly calculated via a look-up table with learnable parameters,
    interacting with queries and keys in self-attention modules.

    This class is different from PyTorch's built-in ``nn.MultiheadAttention`` in:

    1. It supports relative position embedding.
    2. It only supports self attention.
    3. It uses fixed dimension for each head, rather than fixed total dimension.
    """

    def __init__(
        self,
        embed_dim: int | Categorical[int],
        num_heads: int | Categorical[int],
        head_dim: int | None = 64,
        attn_drop: float = 0.,
        proj_drop: float = 0.,
        qkv_bias: bool = False,
        qk_scale: float | None = None,
        rpe: bool = False,
        rpe_length: int = 14,
    ):
        super().__init__()

        # The self. attributes are only used for inspection.
        # The actual values are stored in the submodules.
        if current_model() is not None:
            self.embed_dim = ensure_frozen(embed_dim)
            self.num_heads = ensure_frozen(num_heads)
        else:
            self.embed_dim = embed_dim
            self.num_heads = num_heads

        # head_dim is fixed 64 in official AutoFormer. set head_dim = None to use flex head dim.
        self.head_dim = head_dim or (embed_dim // num_heads)
        self.scale = qk_scale or cast(int, head_dim) ** -0.5
        self.qkv_bias = qkv_bias

        if isinstance(head_dim, Mutable) and isinstance(num_heads, Mutable):
            raise ValueError('head_dim and num_heads can not be both mutable.')

        # Please refer to MixedMultiheadAttention for details.
        self.q = Mutable_KAN([cast(int, embed_dim), cast(int, head_dim) * num_heads], bias=qkv_bias)
        self.k = Mutable_KAN([cast(int, embed_dim), cast(int, head_dim) * num_heads], bias=qkv_bias)
        self.v = Mutable_KAN([cast(int, embed_dim), cast(int, head_dim) * num_heads], bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = Mutable_KAN([cast(int, head_dim) * num_heads, cast(int, embed_dim)])
        self.proj_drop = nn.Dropout(proj_drop)
        self.rpe = rpe

        if self.rpe:
            if isinstance(head_dim, Mutable):
                raise ValueError('head_dim must be a fixed integer when rpe is True.')
            self.rel_pos_embed_k = RelativePosition2D(cast(int, head_dim), rpe_length)
            self.rel_pos_embed_v = RelativePosition2D(cast(int, head_dim), rpe_length)

    def freeze(self, sample) -> RelativePositionSelfAttention:
        new_module = cast(RelativePositionSelfAttention, super().freeze(sample))
        # Handle ad-hoc attributes.
        if isinstance(self.embed_dim, Mutable):
            assert new_module is not self
            new_module.embed_dim = self.embed_dim.freeze(sample)
        if isinstance(self.num_heads, Mutable):
            assert new_module is not self
            new_module.num_heads = self.num_heads.freeze(sample)
        if isinstance(self.head_dim, Mutable):
            assert new_module is not self
            new_module.head_dim = self.head_dim.freeze(sample)
        return new_module

    def forward(self, x):
        B, N, _ = x.shape

        # Infer one of head_dim and num_heads because one of them can be mutable.
        head_dim = -1 if isinstance(self.head_dim, Mutable) else self.head_dim
        num_heads = -1 if isinstance(self.num_heads, Mutable) else self.num_heads

        q = self.q(x).reshape(B, N, num_heads, head_dim).permute(0, 2, 1, 3)
        k = self.k(x).reshape(B, N, num_heads, head_dim).permute(0, 2, 1, 3)
        v = self.v(x).reshape(B, N, num_heads, head_dim).permute(0, 2, 1, 3)
        num_heads, head_dim = q.size(1), q.size(3)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        if self.rpe:
            r_p_k = self.rel_pos_embed_k(N, N)
            attn = attn + (
                q.permute(2, 0, 1, 3).reshape(N, num_heads * B, head_dim) @ r_p_k.transpose(2, 1)
            ).transpose(1, 0).reshape(B, num_heads, N, N) * self.scale

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, num_heads * head_dim)

        if self.rpe:
            attn_1 = attn.permute(2, 0, 1, 3).reshape(N, B * num_heads, N)
            r_p_v = self.rel_pos_embed_v(N, N)
            # The size of attention is (B, num_heads, N, N), reshape it to (N, B*num_heads, N) and do batch matmul with
            # the relative position embedding of V (N, N, head_dim) get shape like (N, B*num_heads, head_dim). We reshape it to the
            # same size as x (B, num_heads, N, hidden_dim)

            x = x + (
                (attn_1 @ r_p_v)
                .transpose(1, 0)
                .reshape(B, num_heads, N, head_dim)
                .transpose(2, 1)
                .reshape(B, N, num_heads * head_dim)
            )

        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def _shape_forward(self, x: ShapeTensor) -> MutableShape:
        assert x.real_shape is not None
        return MutableShape(*x.real_shape)

    def _count_flops(self, x: tuple[MutableShape], y: tuple[MutableShape]) -> FlopsResult:
        """Count the FLOPs of :class:`RelativePositionSelfAttention`.

        RPE module is ignored in this computation.
        """
        _, N, __ = x[0]

        # Dimension working inside.
        interm_dim = self.head_dim * self.num_heads

        params = (
            3 * self.embed_dim * (interm_dim + self.qkv_bias) +    # in_proj
            # skip RPE
            interm_dim * (self.embed_dim + 1)                      # out_proj, bias always true
        )

        flops = (
            N * interm_dim * self.embed_dim * 3 +  # in_proj
            N * N * interm_dim +                   # QK^T
            N * interm_dim * N +                   # RPE (k)
            N * N * interm_dim +                   # AV
            N * interm_dim * N +                   # RPE (v)
            N * interm_dim * self.embed_dim        # out_proj
        )

        return FlopsResult(flops, params)


class TransformerEncoderLayer(nn.Module):
    """
    Multi-head attention + FC + Layer-norm + Dropout.

    Similar to PyTorch's ``nn.TransformerEncoderLayer`` but supports :class:`RelativePositionSelfAttention`.

    Parameters
    ----------
    embed_dim
        Embedding dimension.
    num_heads
        Number of attention heads.
    mlp_ratio
        Ratio of MLP hidden dim to embedding dim.
    drop_path
        Drop path rate.
    drop_rate
        Dropout rate.
    pre_norm
        Whether to apply layer norm before attention.
    **kwargs
        Other arguments for :class:`RelativePositionSelfAttention`.
    """

    def __init__(
        self,
        embed_dim: int | Categorical[int],
        num_heads: int | Categorical[int],
        mlp_ratio: int | float | Categorical[int] | Categorical[float] = 4.,
        drop_path: float = 0.,
        drop_rate: float = 0.,
        pre_norm: bool = True,
        **kwargs
    ):
        super().__init__()

        self.normalize_before = pre_norm

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.attn = RelativePositionSelfAttention(embed_dim=embed_dim, num_heads=num_heads, **kwargs)

        self.attn_layer_norm = MutableLayerNorm(cast(int, embed_dim))
        self.ffn_layer_norm = MutableLayerNorm(cast(int, embed_dim))

        self.activation_fn = nn.GELU()

        self.dropout = nn.Dropout(drop_rate)

        self.fc1 = Mutable_KAN(
            [cast(int, embed_dim),
            cast(int, MutableExpression.to_int(embed_dim * mlp_ratio))]
        )
        self.fc2 = Mutable_KAN(
            [cast(int, MutableExpression.to_int(embed_dim * mlp_ratio)),
            cast(int, embed_dim)]
        )

    def maybe_layer_norm(self, layer_norm, x, before=False, after=False):
        assert before ^ after
        if after ^ self.normalize_before:
            return layer_norm(x)
        else:
            return x

    def forward(self, x):
        """
        Forward function.

        Parameters
        ----------
        x
            Input to the layer of shape ``(batch, patch_num, sample_embed_dim)``.

        Returns
        -------
        Encoded output of shape ``(batch, patch_num, sample_embed_dim)``.
        """
        residual = x
        x = self.maybe_layer_norm(self.attn_layer_norm, x, before=True)
        x = self.attn(x)
        x = self.dropout(x)
        x = self.drop_path(x)
        x = residual + x
        x = self.maybe_layer_norm(self.attn_layer_norm, x, after=True)

        residual = x
        x = self.maybe_layer_norm(self.ffn_layer_norm, x, before=True)
        x = self.fc1(x)
        x = self.activation_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = self.drop_path(x)
        x = residual + x
        x = self.maybe_layer_norm(self.ffn_layer_norm, x, after=True)

        return x


class ClassToken(ParametrizedModule):
    """
    Concat `class token <https://arxiv.org/abs/2010.11929>`__ before patch embedding.

    Parameters
    ----------
    embed_dim
        The dimension of class token.
    """

    def __init__(self, embed_dim: int):
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=.02)

    def forward(self, x):
        return torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)

    def _shape_forward(self, x: ShapeTensor) -> MutableShape:
        assert x.real_shape is not None
        shape = list(x.real_shape)
        return MutableShape(shape[0], shape[1] + 1, shape[2])


class AbsolutePositionEmbedding(ParametrizedModule):
    """Add absolute position embedding on patch embedding."""

    def __init__(self, length: int, embed_dim: int):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.zeros(1, length, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=.02)

    def forward(self, x):
        return x + self.pos_embed

    def _shape_forward(self, x: ShapeTensor) -> MutableShape:
        assert x.real_shape is not None
        return x.real_shape