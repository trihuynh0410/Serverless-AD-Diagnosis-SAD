from torch import nn, einsum, Tensor
from einops import rearrange
from typing import Optional, Tuple, Union, Callable

from nni.nas.nn.pytorch.layers import *

class HSwish(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu6 = MutableReLU6(inplace=True)

    def forward(self, x):
        x = x * self.relu6(x + 3) / 6
        return x


class HSigmoid(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu6 = MutableReLU6(inplace=True)
    
    def forward(self, x):
        x = self.relu6(x + 3) / 6
        return x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = MutableLayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)
        
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.,act_layer=HSwish()):
        super().__init__()
        self.net = nn.Sequential(
            MutableLinear(dim, hidden_dim),
            act_layer,
            MutableDropout(dropout),
            MutableLinear(hidden_dim, dim),
            MutableDropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

# 3D blocks
class Conv3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int, int]],
        stride: Union[int, Tuple[int, int, int]] = 1,
        groups: int = 1,
        padding: str = 'full',
        bias: bool = False,
        use_norm: bool = True,
        act: Optional[Callable] = None,
    ) -> None:
        super().__init__()

        kernel_size = (kernel_size, kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        stride = (stride, stride, stride) if isinstance(stride, int) else stride

        padding = tuple(int((k - 1) / 2) for k in kernel_size) if padding == 'full' else padding

        self.block = nn.Sequential(
            MutableConv3d(in_channels, out_channels, kernel_size, stride, padding=padding, groups=groups, bias=bias),
            MutableBatchNorm3d(out_channels, momentum=0.1) if use_norm else MutableIdentity(),
            act if act is not None else MutableIdentity()
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)

class SepConv3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int, int]],
        stride: Union[int, Tuple[int, int, int]] = 1,
        padding: Union[int, Tuple[int, int, int]] = 0,
        groups: int = 1,
        use_norm_depth: bool = True,
        use_norm_point: bool = True,
        act: Optional[Callable] = None
    )-> None:
        super().__init__()

        self.depthwise = Conv3D(in_channels,
                                in_channels,
                                kernel_size=kernel_size,
                                stride=stride,
                                groups=groups,
                                use_norm=use_norm_depth,
                                act=act)
        self.pointwise = Conv3D(in_channels, out_channels, kernel_size=1, use_norm=use_norm_point)

    def forward(self, x: Tensor) -> Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class SqueezeExcitation(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16, kan=False):
        super().__init__()
        self.se = nn.Sequential(
            MutableAdaptiveAvgPool3d(1),  # Global pooling
            MutableConv3d(in_channels, in_channels // reduction_ratio, kernel_size=1),
            MutableReLU(inplace=True),
            MutableConv3d(in_channels // reduction_ratio, in_channels, kernel_size=1),
            MutableSigmoid()
        )

    def forward(self, x):
        return x * self.se(x)

class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, padding, expand_ratio):
        super().__init__()
        hidden_channels = in_channels * expand_ratio
        self.use_shortcut = stride == 1 and in_channels == out_channels

        # Expansion phase
        layers = []
        if expand_ratio != 1:
            layers.append(Conv3D(in_channels, hidden_channels, kernel_size=1, act=nn.ReLU()))

        # Depthwise convolution phase
        layers.append(SepConv3d(
            hidden_channels, out_channels, kernel_size=3, stride=stride, padding=padding, groups=hidden_channels, act=nn.ReLU()))

        # Squeeze and excitation phase
        layers.append(SqueezeExcitation(out_channels))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_shortcut:
            return x + self.block(x)
        else:
            return self.block(x)