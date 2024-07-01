from typing import Tuple, List, Union, Callable, Optional, cast, Any

import torch, math, nni
from torch import nn

from nni.mutable import MutableExpression
from nni.nas.nn.pytorch import (
    ModelSpace, Repeat,  LayerChoice,
    MutableLinear, MutableLayerNorm, MutableLinear, MutableBatchNorm2d
)
from nni.nas.oneshot.pytorch.supermodule.operation import MixedOperation
from nni.nas.hub.pytorch.proxylessnas import make_divisible, simplify_sequential, ConvBNReLU, DepthwiseSeparableConv

from KANLinear import Mutable_KAN
from ViT import *

MaybeIntChoice = Union[int, MutableExpression]

class UniversialInvertedResidual(nn.Sequential):
    """
    An Universial Inverted Residual Block, originally proposed for the `MobileNetV4 <https://arxiv.org/abs/2404.10518>`.
    It follows a structure of:
        - Optional first depthwise
        - First pointwise
        - Optional second depthwise
        - Second pointwise

    This implementation is the modification of inverted residual block of NNI:

    - https://github.com/microsoft/nni/blob/master/examples/nas/legacy/cream/lib/models/blocks/inverted_residual_block.py#L11

    Parameters
    ----------
    in_channels
        The number of input channels. Can be a value choice.
    out_channels
        The number of output channels. Can be a value choice.
    expand_ratio
        The ratio of intermediate channels with respect to input channels. Can be a value choice.
    kernel_size
        The kernel size of the depthwise convolution. Can be a value choice.
    stride
        The stride of the depthwise convolution.
    squeeze_excite
        Callable to create squeeze and excitation layer. Take hidden channels and input channels as arguments.
    norm_layer
        Callable to create normalization layer. Take input channels as argument.
    activation_layer
        Callable to create activation layer. No input arguments.
    first_conv
        Whether to use the first depthwise convolution
    second_conv
        Whether to use the second depthwise convolution
    """

    def __init__(
        self,
        in_channels: MaybeIntChoice,
        out_channels: MaybeIntChoice,
        expand_ratio: Union[float, MutableExpression[float]],
        kernel_size: MaybeIntChoice = 3,
        stride: int = 1,
        squeeze_excite: Optional[Callable[[MaybeIntChoice, MaybeIntChoice], nn.Module]] = None,
        norm_layer: Optional[Callable[[int], nn.Module]] = None,
        activation_layer: Optional[Callable[..., nn.Module]] = None,
        first_conv: bool = False,
        second_conv: bool = False,
    ) -> None:
        super().__init__()
        self.stride = stride
        self.out_channels = out_channels
        assert stride in [1, 2]
        hidden_ch = cast(int, make_divisible(in_channels * expand_ratio, 8))

        # NOTE: this equivalence check (==) does NOT work for ValueChoice, need to use "is"
        self.has_skip = stride == 1 and in_channels is out_channels

        first_depth = ConvBNReLU(in_channels, in_channels, stride=stride, kernel_size=kernel_size, groups=in_channels,
                       norm_layer=norm_layer, activation_layer=activation_layer) if first_conv else nn.Identity()
        if first_conv and second_conv:
            stride = 1
        Second_depth = ConvBNReLU(hidden_ch, hidden_ch, stride=stride, kernel_size=kernel_size, groups=hidden_ch,
                       norm_layer=norm_layer, activation_layer=activation_layer) if second_conv else nn.Identity()
                
        layers: List[nn.Module] = [
            # first depth-wise
            first_depth,
            # point-wise convolution
            # NOTE: some paper omit this point-wise convolution when stride = 1.
            # In our implementation, if this pw convolution is intended to be omitted,
            # please use SepConv instead.
            ConvBNReLU(in_channels, hidden_ch, kernel_size=1,
                       norm_layer=norm_layer, activation_layer=activation_layer),
            # second depth-wise
            Second_depth,
            # SE
            squeeze_excite(
                cast(int, hidden_ch),
                cast(int, in_channels)
            ) if squeeze_excite is not None else nn.Identity(),
            # pw-linear
            ConvBNReLU(hidden_ch, out_channels, kernel_size=1, norm_layer=norm_layer, activation_layer=nn.Identity),
        ]

        super().__init__(*simplify_sequential(layers))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.has_skip:
            return x + super().forward(x)
        else:
            return super().forward(x)

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
                        op_choices[f'k{kernel_size}_e{exp_ratio}_{mbtype}'] = UniversialInvertedResidual(
                            inp, oup, exp_ratio, kernel_size, stride, first_conv=first_conv, second_conv=second_conv,
                            )

        # It can be implemented with ValueChoice, but we use LayerChoice here
        # to be aligned with the intention of the original ProxylessNAS.
        return LayerChoice(op_choices, label=f'{label}_i{index}')

    return builder

class MobileViT(ModelSpace):
    """
    This is the model space design to classify 3d images by performing operation on 2d slices of those images
    The recommend number of slice is 16 slices per image, with original shape of each 2d slices is 224x224
    
    All 2d slice after layer belongs to ProxylessNAS for position embeded, 
    will be treated as patches as input for AutoFormer parts


    This search space is mixture of two search space:
        - `ProxylessNAS <https://arxiv.org/abs/1812.00332>`__
        - `AutoFormer <https://arxiv.org/abs/2107.00651>`__

    The search space consists of:
        - Four searchable blocks, taken from Univerisal Inverted Residual Blocks
        - Three searchable variables: depth, heads number and MLP ratio.
    
    Parameters
    ----------
    num_labels
        Number of class to classify
    num_slices_per_image
        Number of 2d slices of 3d image, also number of patches
    base_widths
        Widths of each stage, from stem, to body, to head. Length should be 5.
    dropout_rate
        Dropout rate for the final classification layer.
    width_mult
        Width multiplier for the model.
    embed_dim
        expanded_ratio for embedding dim, where embedding dim is 196 x embed_dim
    bn_eps
        Epsilon for batch normalization.
    bn_momentum
        Momentum for batch normalization.
    search_mlp_ratio
        The search space of MLP ratio. Use a list to specify search range.
    search_num_heads
        The search space of number of heads. Use a list to specify search range.
    search_depth
        The search space of depth. Use a list to specify search range.
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
        num_slices_per_image: int = 6,
        base_widths: Tuple[int, ...] = (32, 16, 32, 40, 80),
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
        assert len(base_widths) == 5
        # include the last stage info widths here
        widths = [make_divisible(width * width_mult, 8) for width in base_widths]
        downsamples = [True, False, True, True, False, True]

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
        for stage in range(2, 6):
            # Rather than returning a fixed module here,
            # we return a builder that dynamically creates module for different `repeat_idx`.
            builder = inverted_residual_choice_builder(
                [3, 4], [3, 5], downsamples[stage], widths[stage - 1], widths[stage], f's{stage}')
            if stage < 5:
                blocks.append(Repeat(builder, (1, 3), label=f's{stage}_depth'))
            else:
                # No mutation for depth in the last stage.
                # Directly call builder to initiate one block
                blocks.append(builder(0))

        self.blocks = nn.Sequential(*blocks)

        self.patches_num = num_slices_per_image
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
                rpe_length=int(math.sqrt(self.patches_num)),
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
            "kan":Mutable_KAN([cast(int, embed_dim), num_labels], base_activation=nn.ReLU)
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