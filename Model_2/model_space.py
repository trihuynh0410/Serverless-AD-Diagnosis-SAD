from functools import partial
from typing import Tuple, List, Union, Iterable, Literal, Dict, Callable, Optional, cast, Any

import torch
from torch import nn

import nni
from nni.mutable import MutableExpression, Sample
from nni.nas.hub.pytorch.utils.nn import DropPath
from nni.nas.nn.pytorch import (
    ModelSpace, Repeat,  LayerChoice,
    MutableLinear, MutableLayerNorm, MutableLinear
)
from nni.nas.oneshot.pytorch.supermodule.operation import MixedOperation
from KANLinear import Mutable_KAN
from MobileNetV4 import *
from ViT import *
MaybeIntChoice = Union[int, MutableExpression]

def inverted_residual_choice_builder(
    expand_ratios: List[int],
    kernel_sizes: List[int],
    downsample: bool,
    stage_input_width: int,
    stage_output_width: int,
    label: str
):
    def builder(index):
        stride = 1
        inp = stage_output_width

        if index == 0:
            # first layer in stage
            # do downsample and width reshape
            inp = stage_input_width
            if downsample:
                stride = 2

        oup = stage_output_width
        op_choices = {}
        first_convs = [True, False]
        second_convs = [True, False]
        for first_conv in first_convs:
            for second_conv in second_convs:
                for exp_ratio in expand_ratios:
                    for kernel_size in kernel_sizes:
                        if first_conv and second_conv:
                            mbtype = 'extradw'
                        elif first_conv and not second_conv:
                            mbtype = 'conv_next'
                        elif not first_conv and second_conv:
                            mbtype = 'invert_bottleneck'
                        else:
                            mbtype = 'ffn'
                        if stride == 2 and mbtype == 'ffn':
                            continue
                        op_choices[f'k_{kernel_size}e_{exp_ratio}t_{mbtype}'] = UniversialInvertedResidual(
                            inp, oup, exp_ratio, kernel_size, stride, first_conv=first_conv, second_conv=second_conv
                            )

        # It can be implemented with ValueChoice, but we use LayerChoice here
        # to be aligned with the intention of the original ProxylessNAS.
        return LayerChoice(op_choices, label=f'{label}_i{index}')

    return builder

class MobileViT(ModelSpace):
    """
    The search space that is proposed in `AutoFormer <https://arxiv.org/abs/2107.00651>`__.
    There are four searchable variables: depth, embedding dimension, heads number and MLP ratio.

    Parameters
    ----------
    search_embed_dim
        The search space of embedding dimension. Use a list to specify search range.
    search_mlp_ratio
        The search space of MLP ratio. Use a list to specify search range.
    search_num_heads
        The search space of number of heads. Use a list to specify search range.
    search_depth
        The search space of depth. Use a list to specify search range.
    img_size
        Size of input image.
    patch_size
        Size of image patch.
    in_channels
        Number of channels of the input image.
    num_labels
        Number of classes for classifier.
    qkv_bias
        Whether to use bias item in the qkv embedding.
    drop_rate
        Drop rate of the MLP projection in MSA and FFN.
    attn_drop_rate
        Drop rate of attention.
    drop_path_rate
        Drop path rate.
    pre_norm
        Whether to use pre_norm. Otherwise post_norm is used.
    global_pooling
        Whether to use global pooling to generate the image representation. Otherwise the cls_token is used.
    absolute_position
        Whether to use absolute positional embeddings.
    qk_scale
        The scaler on score map in self-attention.
    rpe
        Whether to use relative position encoding.
    """

    def __init__(
        self,
        num_labels: int = 3,
        base_widths: Tuple[int, ...] = (32, 16, 32, 40),
        dropout_rate: float = 0.,
        width_mult: float = 1.0,
        bn_eps: float = 1e-3,
        bn_momentum: float = 0.1,
        embed_dim: int = 2,
        search_mlp_ratio: Tuple[float, ...] = (3.0, 3.5, 4.0),
        search_num_heads: Tuple[int, ...] = (3, 4),
        search_depth: Tuple[int, ...] = (12, 13, 14),
        qkv_bias: bool = True,
        attn_drop_rate: float = 0.,
        drop_path_rate: float = 0.1,
        pre_norm: bool = True,
        absolute_position: bool = True,
        qk_scale: float | None = None,
        rpe: bool = True,
    ):
        super().__init__()
        assert len(base_widths) == 4
        # include the last stage info widths here
        widths = [make_divisible(width * width_mult, 8) for width in base_widths]
        downsamples = [True, True, True, True, False, True]

        depth = nni.choice("depth", list(search_depth))
        mlp_ratios = [nni.choice(f"mlp_ratio_{i}", list(search_mlp_ratio)) for i in range(max(search_depth))]
        num_heads = [nni.choice(f"num_head_{i}", list(search_num_heads)) for i in range(max(search_depth))]

        widths.append(embed_dim)
        self.num_labels = num_labels
        self.dropout_rate = dropout_rate
        self.bn_eps = bn_eps
        self.bn_momentum = bn_momentum

        self.stem = ConvBNReLU(1, widths[0], stride=2, norm_layer=MutableBatchNorm2d)

        blocks: List[nn.Module] = [
            # first stage is fixed
            DepthwiseSeparableConv(widths[0], widths[1], kernel_size=3, stride=1)
        ]

        # https://github.com/ultmaster/AceNAS/blob/46c8895fd8a05ffbc61a6b44f1e813f64b4f66b7/searchspace/proxylessnas/__init__.py#L21
        for stage in range(2, 5):
            # Rather than returning a fixed module here,
            # we return a builder that dynamically creates module for different `repeat_idx`.
            builder = inverted_residual_choice_builder(
                [3, 4], [3, 5], downsamples[stage], widths[stage - 1], widths[stage], f's{stage}')
            if stage < 4:
                blocks.append(Repeat(builder, (1, 3), label=f's{stage}_depth'))
            else:
                # No mutation for depth in the last stage.
                # Directly call builder to initiate one block
                blocks.append(builder(0))

        self.blocks = nn.Sequential(*blocks)

        self.patches_num = 6
        embed_dim = embed_dim*14*14
        self.cls_token = ClassToken(cast(int, embed_dim))
        self.pos_embed = AbsolutePositionEmbedding(self.patches_num + 1, cast(int, embed_dim)) if absolute_position else nn.Identity()

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, max(search_depth))]  # stochastic depth decay rule

        self.transformers = Repeat(
            lambda index: TransformerEncoderLayer(
                embed_dim=embed_dim,
                num_heads=num_heads[index],
                mlp_ratio=mlp_ratios[index],
                qkv_bias=qkv_bias,
                drop_rate=dropout_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[index],
                rpe_length=2,
                qk_scale=qk_scale,
                rpe=rpe,
                pre_norm=pre_norm
            ), depth
        )

        self.norm = MutableLayerNorm(cast(int, embed_dim)) if pre_norm else nn.Identity()
        self.dropout_layer = nn.Dropout(dropout_rate)

        self.head = MutableLinear(cast(int, embed_dim), num_labels) if num_labels > 0 else nn.Identity()
        self.head = LayerChoice({
            "mlp":MutableLinear(cast(int, embed_dim), num_labels),
            "kan":Mutable_KAN([cast(int, embed_dim), num_labels], base_activation=nn.Softmax)
        }, label='head')
    @classmethod
    def extra_oneshot_hooks(cls, strategy):
        # General hooks agnostic to strategy.
        return [MixedAbsolutePositionEmbedding.mutate, MixedClassToken.mutate]

    def forward(self, x):
        num_images, num_slices_per_image, _, height, width = x.size()

        x = self.stem(x.view(-1, 1, height, width))
        x = self.blocks(x)
        x = x.view(num_images, num_slices_per_image, x.size(1)*x.size(2)*x.size(3))
        x = self.cls_token(x)
        x = self.pos_embed(x)
        x = self.transformers(x)
        x = self.norm(x)
        x = torch.mean(x[:, 1:], dim=1)
        x = self.dropout_layer(x)
        x = self.head(x)

        return x

# one-shot implementations

class MixedAbsolutePositionEmbedding(MixedOperation, AbsolutePositionEmbedding):
    """ Mixed absolute position embedding add operation.

    Supported arguments are:

    - ``embed_dim``

    Prefix of pos_embed will be sliced.
    """
    bound_type = AbsolutePositionEmbedding
    argument_list = ['embed_dim']

    def super_init_argument(self, name: str, value_choice: MutableExpression):
        return max(value_choice.grid())

    def freeze_weight(self, embed_dim, **kwargs) -> Any:
        from nni.nas.oneshot.pytorch.supermodule._operation_utils import Slicable, MaybeWeighted
        embed_dim_ = MaybeWeighted(embed_dim)
        pos_embed = Slicable(self.pos_embed)[..., :embed_dim_]

        return {'pos_embed': pos_embed}

    def forward_with_args(self, embed_dim,
                          inputs: torch.Tensor) -> torch.Tensor:
        pos_embed = self.freeze_weight(embed_dim)['pos_embed']
        assert isinstance(pos_embed, torch.Tensor)

        return inputs + pos_embed


class MixedClassToken(MixedOperation, ClassToken):
    """Mixed class token concat operation.

    Supported arguments are:

    - ``embed_dim``

    Prefix of cls_token will be sliced.
    """
    bound_type = ClassToken
    argument_list = ['embed_dim']

    def super_init_argument(self, name: str, value_choice: MutableExpression):
        return max(value_choice.grid())

    def freeze_weight(self, embed_dim, **kwargs) -> Any:
        from nni.nas.oneshot.pytorch.supermodule._operation_utils import Slicable, MaybeWeighted
        embed_dim_ = MaybeWeighted(embed_dim)
        cls_token = Slicable(self.cls_token)[..., :embed_dim_]

        return {'cls_token': cls_token}

    def forward_with_args(self, embed_dim,
                          inputs: torch.Tensor) -> torch.Tensor:
        cls_token = self.freeze_weight(embed_dim)['cls_token']
        assert isinstance(cls_token, torch.Tensor)

        return torch.cat((cls_token.expand(inputs.shape[0], -1, -1), inputs), dim=1)