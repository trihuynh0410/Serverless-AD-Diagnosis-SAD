from typing import List, Union, Literal, Callable, Optional, cast, Type, Iterator

import torch
from torch import nn

from nni.mutable import MutableExpression
from nni.nas.nn.pytorch import LayerChoice, MutableBatchNorm2d, MutableConv2d
MaybeIntChoice = Union[int, MutableExpression]

def make_divisible(v: Union[MutableExpression[int], MutableExpression[float], int, float], divisor, min_val=None) -> MaybeIntChoice:
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_val is None:
        min_val = divisor
    # This should work for both value choices and constants.
    new_v = MutableExpression.max(min_val, round(v + divisor // 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    return MutableExpression.condition(new_v < 0.9 * v, new_v + divisor, new_v)

def simplify_sequential(sequentials: List[nn.Module]) -> Iterator[nn.Module]:
    """
    Flatten the sequential blocks so that the hierarchy looks better.
    Eliminate identity modules automatically.
    """
    for module in sequentials:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                # no recursive expansion
                if not isinstance(submodule, nn.Identity):
                    yield submodule
        else:
            if not isinstance(module, nn.Identity):
                yield module

class SqueezeExcite(nn.Module):
    """Squeeze-and-excite layer.

    We can't use the op from ``torchvision.ops`` because it's not (yet) properly wrapped,
    and ValueChoice couldn't be processed.

    Reference:

    - https://github.com/rwightman/pytorch-image-models/blob/b7cb8d03/timm/models/efficientnet_blocks.py#L26
    - https://github.com/d-li14/mobilenetv3.pytorch/blob/3e6938cedcbbc5ee5bc50780ea18e644702d85fc/mobilenetv3.py#L53
    """

    def __init__(self,
                 channels: int,
                 reduction_ratio: float = 0.25,
                 gate_layer: Optional[Callable[..., nn.Module]] = None,
                 activation_layer: Optional[Callable[..., nn.Module]] = None):
        super().__init__()

        rd_channels = make_divisible(channels * reduction_ratio, 8)
        gate_layer = gate_layer or nn.Hardsigmoid
        activation_layer = activation_layer or nn.ReLU
        self.conv_reduce = MutableConv2d(channels, rd_channels, 1, bias=True)
        self.act1 = activation_layer(inplace=True)
        self.conv_expand = MutableConv2d(rd_channels, channels, 1, bias=True)
        self.gate = gate_layer()

    def forward(self, x):
        x_se = x.mean((2, 3), keepdim=True)
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        return x * self.gate(x_se)


def _se_or_skip(hidden_ch: int, input_ch: int, optional: bool, se_from_exp: bool, label: str) -> nn.Module:
    ch = hidden_ch if se_from_exp else input_ch
    if optional:
        return LayerChoice({
            'identity': nn.Identity(),
            'se': SqueezeExcite(ch)
        }, label=label)
    else:
        return SqueezeExcite(ch)

class ConvBNReLU(nn.Sequential):
    """
    The template for a conv-bn-relu block.
    """

    def __init__(
        self,
        in_channels: MaybeIntChoice,
        out_channels: MaybeIntChoice,
        kernel_size: MaybeIntChoice = 3,
        stride: int = 1,
        groups: MaybeIntChoice = 1,
        norm_layer: Optional[Callable[[int], nn.Module]] = None,
        activation_layer: Optional[Callable[..., nn.Module]] = None,
        dilation: int = 1,
    ) -> None:
        padding = (kernel_size - 1) // 2 * dilation
        if norm_layer is None:
            norm_layer = MutableBatchNorm2d
        if activation_layer is None:
            activation_layer = nn.ReLU6
        # If no normalization is used, set bias to True
        # https://github.com/google-research/google-research/blob/20736344/tunas/rematlib/mobile_model_v3.py#L194
        norm = norm_layer(cast(int, out_channels))
        no_normalization = isinstance(norm, nn.Identity)
        blocks: List[nn.Module] = [
            MutableConv2d(
                cast(int, in_channels),
                cast(int, out_channels),
                cast(int, kernel_size),
                stride,
                cast(int, padding),
                dilation=dilation,
                groups=cast(int, groups),
                bias=no_normalization
            ),
            # Normalization, regardless of batchnorm or identity
            norm,
            # One pytorch implementation as an SE here, to faithfully reproduce paper
            # We follow a more accepted approach to put SE outside
            # Reference: https://github.com/d-li14/mobilenetv3.pytorch/issues/18
            activation_layer(inplace=True)
        ]

        super().__init__(*simplify_sequential(blocks))

class UniversialInvertedResidual(nn.Sequential):
    """
    An Inverted Residual Block, sometimes called an MBConv Block, is a type of residual block used for image models
    that uses an inverted structure for efficiency reasons.

    It was originally proposed for the `MobileNetV2 <https://arxiv.org/abs/1801.04381>`__ CNN architecture.
    It has since been reused for several mobile-optimized CNNs.
    It follows a narrow -> wide -> narrow approach, hence the inversion.
    It first widens with a 1x1 convolution, then uses a 3x3 depthwise convolution (which greatly reduces the number of parameters),
    then a 1x1 convolution is used to reduce the number of channels so input and output can be added.

    This implementation is sort of a mixture between:

    - https://github.com/google-research/google-research/blob/20736344/tunas/rematlib/mobile_model_v3.py#L453
    - https://github.com/rwightman/pytorch-image-models/blob/b7cb8d03/timm/models/efficientnet_blocks.py#L134

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
            first_depth,
            # point-wise convolution
            # NOTE: some paper omit this point-wise convolution when stride = 1.
            # In our implementation, if this pw convolution is intended to be omitted,
            # please use SepConv instead.
            ConvBNReLU(in_channels, hidden_ch, kernel_size=1,
                       norm_layer=norm_layer, activation_layer=activation_layer),
            # depth-wise
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