from functools import partial
from typing import Tuple, List, Union, Iterable, Literal, Dict, Callable, Optional, cast, Type, Iterator

import torch
from torch import nn

import nni
from nni.mutable import MutableExpression, Sample
from nni.nas.oneshot.pytorch.supermodule.sampling import PathSamplingRepeat
from nni.nas.oneshot.pytorch.supermodule.differentiable import DifferentiableMixedRepeat
from nni.nas.nn.pytorch import ModelSpace, LayerChoice, Repeat, Cell, MutableConv2d, MutableBatchNorm2d, MutableLinear, model_context
from nni.nas.hub.pytorch.utils.nn import DropPath

from kan import *

MaybeIntChoice = Union[int, MutableExpression]

OPS = {
    'none': lambda C, stride, affine:
        Zero(stride),
    'avg_pool_3x3': lambda C, stride, affine:
        nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False),
    'max_pool_3x3': lambda C, stride, affine:
        nn.MaxPool2d(3, stride=stride, padding=1),
    'skip_connect': lambda C, stride, affine:
        nn.Identity() if stride == 1 else FactorizedReduce(C, C, affine=affine),
    'conv_1x1': lambda C, stride, affine:
        nn.Sequential(
            nn.ReLU(inplace=False),
            MutableConv2d(C, C, 1, stride=stride, padding=0, bias=False),
            MutableBatchNorm2d(C, affine=affine)
        ),
    'conv_3x3': lambda C, stride, affine:
        nn.Sequential(
            nn.ReLU(inplace=False),
            MutableConv2d(C, C, 3, stride=stride, padding=1, bias=False),
            MutableBatchNorm2d(C, affine=affine)
        ),
    'sep_conv_3x3': lambda C, stride, affine:
        SepConv(C, C, 3, stride, 1, affine=affine),
    'sep_conv_5x5': lambda C, stride, affine:
        SepConv(C, C, 5, stride, 2, affine=affine),
    'dil_conv_3x3': lambda C, stride, affine:
        DilConv(C, C, 3, stride, 2, 2, affine=affine),
    'dil_conv_5x5': lambda C, stride, affine:
        DilConv(C, C, 5, stride, 4, 2, affine=affine),
    'kan_hswish': lambda C, stride, affine:
        KanWarapper(C,C,base_activation=nn.Hardswish),
    'kan_relu6': lambda C, stride, affine:
        KanWarapper(C,C,base_activation=nn.ReLU6),
    'kan_silu': lambda C, stride, affine:
        KanWarapper(C,C,base_activation=nn.SiLU),    
    'extra_dw': lambda C, stride, affine:
        UniversialInvertedResidual(
            C,C,3,3, stride,
            squeeze_excite=cast(Callable[[MaybeIntChoice, MaybeIntChoice], nn.Module], 
                partial(_se_or_skip, optional=False, se_from_exp=True, label=f'extra_dw')),
            first_conv=True, second_conv=True
                ),
    'invert_bottleneck': lambda C, stride, affine:
        UniversialInvertedResidual(
            C,C,3,3, stride,
            squeeze_excite=cast(Callable[[MaybeIntChoice, MaybeIntChoice], nn.Module], 
                partial(_se_or_skip, optional=False, se_from_exp=True, label=f'ib')),
            first_conv=False, second_conv=True
                ),
    'conv_next': lambda C, stride, affine:
        UniversialInvertedResidual(
            C,C,3,3, stride,
            squeeze_excite=cast(Callable[[MaybeIntChoice, MaybeIntChoice], nn.Module], 
                partial(_se_or_skip, optional=False, se_from_exp=True, label=f'conv_next')),
            first_conv=True, second_conv=False
                ),
    'ffn': lambda C, stride, affine:
        UniversialInvertedResidual(
            C,C,3,3, stride,
            squeeze_excite=cast(Callable[[MaybeIntChoice, MaybeIntChoice], nn.Module], 
                partial(_se_or_skip, optional=False, se_from_exp=True, label=f'ffn')),
            first_conv=False, second_conv=False
                ), 
   
}

class KanWarapper(nn.Module):
    def __init__(self, in_channel, out_channel, base_activation):
        super().__init__()

        self.proj_func = MutableKAN([in_channel, in_channel//2, out_channel], base_activation=base_activation)

    def forward(self, x):
        x = self.to_last_dim(x)
        x = self.proj_func(x)
        x = self.to_first_dim(x)
        return x

    @staticmethod
    def to_last_dim(t):
        num_dims = len(t.shape)
        permute_order = [0] + list(range(2, num_dims)) + [1]
        return t.permute(*permute_order)
    
    @staticmethod
    def to_first_dim(t):
        num_dims = len(t.shape)
        permute_order = [0, num_dims-1] + list(range(1, num_dims-1))
        return t.permute(*permute_order)

class ReLUConvBN(nn.Sequential):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super().__init__(
            nn.ReLU(inplace=False),
            MutableConv2d(
                C_in, C_out, kernel_size, stride=stride,
                padding=padding, bias=False
            ),
            MutableBatchNorm2d(C_out, affine=affine)
        )


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


def _act_fn(act_alias: Literal['hswish', 'swish', 'relu']) -> Type[nn.Module]:
    if act_alias == 'hswish':
        return nn.Hardswish
    elif act_alias == 'swish':
        return nn.SiLU
    elif act_alias == 'relu':
        return nn.ReLU
    else:
        raise ValueError(f'Unsupported act alias: {act_alias}')

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

class Zero(nn.Module):

    def __init__(self, stride):
        super().__init__()
        self.stride = stride

    def forward(self, x):
        if self.stride == 1:
            return x.mul(0.)
        return x[:, :, ::self.stride, ::self.stride].mul(0.)

class FactorizedReduce(nn.Module):

    def __init__(self, C_in, C_out, affine=True):
        super().__init__()
        if isinstance(C_out, int):
            assert C_out % 2 == 0
        else:   # is a value choice
            assert all(c % 2 == 0 for c in C_out.grid())
        self.relu = nn.ReLU(inplace=False)
        self.conv_1 = MutableConv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.conv_2 = MutableConv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.bn = MutableBatchNorm2d(C_out, affine=affine)
        self.pad = nn.ConstantPad2d((0, 1, 0, 1), 0)

    def forward(self, x):
        x = self.relu(x)
        y = self.pad(x)
        out = torch.cat([self.conv_1(x), self.conv_2(y[:, :, 1:, 1:])], dim=1)
        out = self.bn(out)
        return out
    
class DilConv(nn.Sequential):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True):
        super().__init__(
            nn.ReLU(inplace=False),
            MutableConv2d(
                C_in, C_in, kernel_size=kernel_size, stride=stride,
                padding=padding, dilation=dilation, groups=C_in, bias=False
            ),
            MutableConv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            MutableBatchNorm2d(C_out, affine=affine),
        )


class SepConv(nn.Sequential):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super().__init__(
            nn.ReLU(inplace=False),
            MutableConv2d(
                C_in, C_in, kernel_size=kernel_size, stride=stride,
                padding=padding, groups=C_in, bias=False
            ),
            MutableConv2d(C_in, C_in, kernel_size=1, padding=0, bias=False),
            MutableBatchNorm2d(C_in, affine=affine),
            nn.ReLU(inplace=False),
            MutableConv2d(
                C_in, C_in, kernel_size=kernel_size, stride=1,
                padding=padding, groups=C_in, bias=False
            ),
            MutableConv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            MutableBatchNorm2d(C_out, affine=affine),
        )

class AuxiliaryHead(nn.Module):
    def __init__(self, C: int, num_labels: int, dataset: Literal['imagenet', 'cifar']):
        super().__init__()
        if dataset == 'imagenet':
            # assuming input size 14x14
            stride = 2
        elif dataset == 'cifar':
            stride = 3

        self.features = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.AvgPool2d(5, stride=stride, padding=0, count_include_pad=False),
            MutableConv2d(C, 128, 1, bias=False),
            MutableBatchNorm2d(128),
            nn.ReLU(inplace=True),
            MutableConv2d(128, 768, 2, bias=False),
            MutableBatchNorm2d(768),
            nn.ReLU(inplace=True)
        )
        self.classifier = MutableLinear(768, num_labels)

    def forward(self, x, num_images, num_slices_per_image):
            x = self.features(x)
            #
            # x = self.classifier(x.view(x.size(0), -1))
            
            x = x.view(num_images, num_slices_per_image, -1).mean(1)  # Reshape and average over slices

            x = self.classifier(x)
            return x
    

class CellPreprocessor(nn.Module):
    """
    Aligning the shape of predecessors.

    If the last cell is a reduction cell, ``pre0`` should be ``FactorizedReduce`` instead of ``ReLUConvBN``.
    See :class:`CellBuilder` on how to calculate those channel numbers.
    """

    def __init__(self, C_pprev: MaybeIntChoice, C_prev: MaybeIntChoice, C: MaybeIntChoice, last_cell_reduce: bool) -> None:
        super().__init__()

        if last_cell_reduce:
            self.pre0 = FactorizedReduce(cast(int, C_pprev), cast(int, C))
        else:
            self.pre0 = ReLUConvBN(cast(int, C_pprev), cast(int, C), 1, 1, 0)
        self.pre1 = ReLUConvBN(cast(int, C_prev), cast(int, C), 1, 1, 0)

    def forward(self, cells):
        assert len(cells) == 2
        pprev, prev = cells
        pprev = self.pre0(pprev)
        prev = self.pre1(prev)

        return [pprev, prev]

class CellPostprocessor(nn.Module):
    """
    The cell outputs previous cell + this cell, so that cells can be directly chained.
    """

    def forward(self, this_cell, previous_cells):
        return [previous_cells[-1], this_cell]


class CellBuilder:
    """The cell builder is used in Repeat.
    Builds an cell each time it's "called".
    Note that the builder is ephemeral, it can only be called once for every index.
    """

    def __init__(self, op_candidates: List[str],
                 C_prev_in: MaybeIntChoice,
                 C_in: MaybeIntChoice,
                 C: MaybeIntChoice,
                 num_nodes: int,
                 merge_op: Literal['all', 'loose_end'],
                 first_cell_reduce: bool, last_cell_reduce: bool,
                 drop_path_prob: float):
        self.C_prev_in = C_prev_in      # This is the out channels of the cell before last cell.
        self.C_in = C_in                # This is the out channesl of last cell.
        self.C = C                      # This is NOT C_out of this stage, instead, C_out = C * len(cell.output_node_indices)
        self.op_candidates = op_candidates
        self.num_nodes = num_nodes
        self.merge_op: Literal['all', 'loose_end'] = merge_op
        self.first_cell_reduce = first_cell_reduce
        self.last_cell_reduce = last_cell_reduce
        self.drop_path_prob = drop_path_prob
        self._expect_idx = 0

        # It takes an index that is the index in the repeat.
        # Number of predecessors for each cell is fixed to 2.
        self.num_predecessors = 2

        # Number of ops per node is fixed to 2.
        self.num_ops_per_node = 2

    def op_factory(self, node_index: int, op_index: int, input_index: Optional[int], *,
                   op: str, channels: int, is_reduction_cell: bool):
        if is_reduction_cell and (
            input_index is None or input_index < self.num_predecessors
        ):  # could be none when constructing search space
            stride = 2
        else:
            stride = 1
        operation = OPS[op](channels, stride, True)
        if self.drop_path_prob > 0 and not isinstance(operation, nn.Identity):
            # Omit drop-path when operation is skip connect.
            # https://github.com/quark0/darts/blob/f276dd346a09ae3160f8e3aca5c7b193fda1da37/cnn/model.py#L54
            return nn.Sequential(operation, DropPath(self.drop_path_prob))
        return operation

    def __call__(self, repeat_idx: int):
        if self._expect_idx != repeat_idx:
            raise ValueError(f'Expect index {self._expect_idx}, found {repeat_idx}')

        # Reduction cell means stride = 2 and channel multiplied by 2.
        is_reduction_cell = repeat_idx == 0 and self.first_cell_reduce

        # self.C_prev_in, self.C_in, self.last_cell_reduce are updated after each cell is built.
        preprocessor = CellPreprocessor(self.C_prev_in, self.C_in, self.C, self.last_cell_reduce)

        ops_factory: Dict[str, Callable[[int, int, Optional[int]], nn.Module]] = {}
        for op in self.op_candidates:
            if is_reduction_cell and (op == 'kan_hswish' or op =='kan_relu6' or op == 'kan_silu' or op == 'ffn'):
                continue
            ops_factory[op] = partial(self.op_factory, op=op, channels=cast(int, self.C), is_reduction_cell=is_reduction_cell)

        cell = Cell(ops_factory, self.num_nodes, self.num_ops_per_node, self.num_predecessors, self.merge_op,
                    preprocessor=preprocessor, postprocessor=CellPostprocessor(),
                    label='reduce' if is_reduction_cell else 'normal')

        # update state
        self.C_prev_in = self.C_in
        self.C_in = self.C * len(cell.output_node_indices)
        self.last_cell_reduce = is_reduction_cell
        self._expect_idx += 1

        return cell

class NDSStage(Repeat):
    """This class defines NDSStage, a special type of Repeat, for isinstance check, and shape alignment.

    In NDS, we can't simply use Repeat to stack the blocks,
    because the output shape of each stacked block can be different.
    This is a problem for one-shot strategy because they assume every possible candidate
    should return values of the same shape.

    Therefore, we need :class:`NDSStagePathSampling` and :class:`NDSStageDifferentiable`
    to manually align the shapes -- specifically, to transform the first block in each stage.

    This is not required though, when depth is not changing, or the mutable depth causes no problem
    (e.g., when the minimum depth is large enough).

    .. attention::

       Assumption: Loose end is treated as all in ``merge_op`` (the case in one-shot),
       which enforces reduction cell and normal cells in the same stage to have the exact same output shape.
    """

    estimated_out_channels_prev: int
    """Output channels of cells in last stage."""

    estimated_out_channels: int
    """Output channels of this stage. It's **estimated** because it assumes ``all`` as ``merge_op``."""

    downsampling: bool
    """This stage has downsampling"""

    def first_cell_transformation_factory(self) -> Optional[nn.Module]:
        """To make the "previous cell" in first cell's output have the same shape as cells in this stage."""
        if self.downsampling:
            return FactorizedReduce(self.estimated_out_channels_prev, self.estimated_out_channels)
        elif self.estimated_out_channels_prev is not self.estimated_out_channels:
            # Can't use != here, ValueChoice doesn't support
            return ReLUConvBN(self.estimated_out_channels_prev, self.estimated_out_channels, 1, 1, 0)
        return None


class NDSStagePathSampling(PathSamplingRepeat):
    """The path-sampling implementation (for one-shot) of each NDS stage if depth is mutating."""
    @classmethod
    def mutate(cls, module, name, memo, mutate_kwargs):
        if isinstance(module, NDSStage) and isinstance(module.depth_choice, MutableExpression):
            return cls(
                module.first_cell_transformation_factory(),
                list(module.blocks),
                module.depth_choice
            )

    def __init__(self, first_cell_transformation: Optional[nn.Module], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.first_cell_transformation = first_cell_transformation

    def _reduction(self, items: List[Tuple[torch.Tensor, torch.Tensor]], sampled: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        if 1 not in sampled or self.first_cell_transformation is None:
            return super()._reduction(items, sampled)
        # items[0] must be the result of first cell
        assert len(items[0]) == 2
        # Only apply the transformation on "prev" output.
        items[0] = (self.first_cell_transformation(items[0][0]), items[0][1])
        return super()._reduction(items, sampled)


class NDSStageDifferentiable(DifferentiableMixedRepeat):
    """The differentiable implementation (for one-shot) of each NDS stage if depth is mutating."""
    @classmethod
    def mutate(cls, module, name, memo, mutate_kwargs):
        if isinstance(module, NDSStage) and isinstance(module.depth_choice, MutableExpression):
            # Only interesting when depth is mutable
            softmax = mutate_kwargs.get('softmax', nn.Softmax(-1))
            alphas = {}
            for label in module.depth_choice.simplify():
                alphas[label] = memo[label]
            return cls(
                module.first_cell_transformation_factory(),
                list(module.blocks),
                module.depth_choice,
                softmax,
                alphas
            )

    def __init__(self, first_cell_transformation: Optional[nn.Module], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.first_cell_transformation = first_cell_transformation

    def _reduction(
        self, items: List[Tuple[torch.Tensor, torch.Tensor]], weights: List[float], depths: List[int]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if 1 not in depths or self.first_cell_transformation is None:
            return super()._reduction(items, weights, depths)
        # Same as NDSStagePathSampling
        assert len(items[0]) == 2
        items[0] = (self.first_cell_transformation(items[0][0]), items[0][1])
        return super()._reduction(items, weights, depths)


_INIT_PARAMETER_DOCS = """

    Notes
    -----

    To use NDS spaces with one-shot strategies,
    especially when depth is mutating (i.e., ``num_cells`` is set to a tuple / list),
    please use :class:`~nni.nas.hub.pytorch.nasnet.NDSStagePathSampling` (with ENAS and RandomOneShot)
    and :class:`~nni.nas.hub.pytorch.nasnet.NDSStageDifferentiable` (with DARTS and Proxyless) into ``mutation_hooks``.
    This is because the output shape of each stacked block in :class:`~nni.nas.hub.pytorch.nasnet.NDSStage` can be different.
    For example::

        from nni.nas.hub.pytorch.nasnet import NDSStageDifferentiable
        darts_strategy = strategy.DARTS(mutation_hooks=[NDSStageDifferentiable.mutate])

    Parameters
    ----------
    width
        A fixed initial width or a tuple of widths to choose from.
    num_cells
        A fixed number of cells (depths) to stack, or a tuple of depths to choose from.
    dataset
        The essential differences are in "stem" cells, i.e., how they process the raw image input.
        Choosing "imagenet" means more downsampling at the beginning of the network.
    auxiliary_loss
        If true, another auxiliary classification head will produce the another prediction.
        This makes the output of network two logits in the training phase.
    drop_path_prob
        Apply drop path. Enabled when it's set to be greater than 0.

"""

class NDS(ModelSpace):
    __doc__ = """
    The unified version of NASNet search space.

    We follow the implementation in
    `unnas <https://github.com/facebookresearch/unnas/blob/main/pycls/models/nas/nas.py>`__.
    See `On Network Design Spaces for Visual Recognition <https://arxiv.org/abs/1905.13214>`__ for details.

    Different NAS papers usually differ in the way that they specify ``op_candidates`` and ``merge_op``.
    ``dataset`` here is to give a hint about input resolution, so as to create reasonable stem and auxiliary heads.

    NDS has a speciality that it has mutable depths/widths.
    This is implemented by accepting a list of int as ``num_cells`` / ``width``.
    """ + _INIT_PARAMETER_DOCS.rstrip() + """
    op_candidates
        List of operator candidates. Must be from ``OPS``.
    merge_op
        See :class:`~nni.nas.nn.pytorch.Cell`.
    num_nodes_per_cell
        See :class:`~nni.nas.nn.pytorch.Cell`.
    """

    def __init__(self,
                 op_candidates: List[str],
                 merge_op: Literal['all', 'loose_end'] = 'all',
                 num_nodes_per_cell: int = 4,
                 width: Union[Tuple[int, ...], int] = 16,
                 num_cells: Union[Tuple[int, ...], int] = 20,
                 dataset: Literal['cifar', 'imagenet'] = 'imagenet',
                 auxiliary_loss: bool = False,
                 drop_path_prob: float = 0.):
        super().__init__()

        self.op_candidates = op_candidates
        self.merge_op = merge_op
        self.num_nodes_per_cell = num_nodes_per_cell
        self.width = width
        self.num_cells = num_cells
        self.dataset = dataset
        self.num_labels = 3
        self.auxiliary_loss = auxiliary_loss
        self.drop_path_prob = drop_path_prob
        self.num_slices_per_image = 6
        # preprocess the specified width and depth
        if isinstance(width, Iterable):
            C = nni.choice('width', list(width))
        else:
            C = width

        self.num_cells: MaybeIntChoice = cast(int, num_cells)
        if isinstance(num_cells, Iterable):
            self.num_cells = nni.choice('depth', list(num_cells))
        num_cells_per_stage = [(i + 1) * self.num_cells // 3 - i * self.num_cells // 3 for i in range(3)]

        # auxiliary head is different for network targetted at different datasets
        if dataset == 'imagenet':
            self.stem0 = nn.Sequential(
                MutableConv2d(1, cast(int, C // 2), kernel_size=3, stride=2, padding=1, bias=False),
                MutableBatchNorm2d(cast(int, C // 2)),
                nn.ReLU(inplace=True),
                MutableConv2d(cast(int, C // 2), cast(int, C), 3, stride=2, padding=1, bias=False),
                MutableBatchNorm2d(C),
            )
            self.stem1 = nn.Sequential(
                nn.ReLU(inplace=True),
                MutableConv2d(cast(int, C), cast(int, C), 3, stride=2, padding=1, bias=False),
                MutableBatchNorm2d(C),
            )
            C_pprev = C_prev = C_curr = C
            last_cell_reduce = True
        elif dataset == 'cifar':
            self.stem = nn.Sequential(
                MutableConv2d(1, cast(int, 3 * C), 3, padding=1, bias=False),
                MutableBatchNorm2d(cast(int, 3 * C))
            )
            C_pprev = C_prev = 3 * C
            C_curr = C
            last_cell_reduce = False
        else:
            raise ValueError(f'Unsupported dataset: {dataset}')

        self.stages = nn.ModuleList()
        for stage_idx in range(3):
            if stage_idx > 0:
                C_curr *= 2
            # For a stage, we get C_in, C_curr, and C_out.
            # C_in is only used in the first cell.
            # C_curr is number of channels for each operator in current stage.
            # C_out is usually `C * num_nodes_per_cell` because of concat operator.
            cell_builder = CellBuilder(op_candidates, C_pprev, C_prev, C_curr, num_nodes_per_cell,
                                       merge_op, stage_idx > 0, last_cell_reduce, drop_path_prob)
            stage: Union[NDSStage, nn.Sequential] = NDSStage(cell_builder, num_cells_per_stage[stage_idx])

            if isinstance(stage, NDSStage):
                stage.estimated_out_channels_prev = cast(int, C_prev)
                stage.estimated_out_channels = cast(int, C_curr * num_nodes_per_cell)
                stage.downsampling = stage_idx > 0

            self.stages.append(stage)

            # NOTE: output_node_indices will be computed on-the-fly in trial code.
            # When constructing model space, it's just all the nodes in the cell,
            # which happens to be the case of one-shot supernet.

            # C_pprev is output channel number of last second cell among all the cells already built.
            if len(stage) > 1:
                # Contains more than one cell
                C_pprev = len(cast(Cell, stage[-2]).output_node_indices) * C_curr
            else:
                # Look up in the out channels of last stage.
                C_pprev = C_prev

            # This was originally,
            # C_prev = num_nodes_per_cell * C_curr.
            # but due to loose end, it becomes,
            C_prev = len(cast(Cell, stage[-1]).output_node_indices) * C_curr

            # Useful in aligning the pprev and prev cell.
            last_cell_reduce = cell_builder.last_cell_reduce

            if stage_idx == 2:
                C_to_auxiliary = C_prev

        if auxiliary_loss:
            assert isinstance(self.stages[2], nn.Sequential), 'Auxiliary loss can only be enabled in retrain mode.'
            self.auxiliary_head = AuxiliaryHead(C_to_auxiliary, self.num_labels, dataset=self.dataset)  # type: ignore


        self.global_pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = LayerChoice({
            "mlp":MutableLinear(cast(int, C_prev), self.num_labels),
            "kan":KanWarapper(cast(int, C_prev), self.num_labels,base_activation=nn.Softmax)
        }, label='classifier')
        # self.classifier1 = LayerChoice({
        #     "mlp_1":MutableLinear(cast(int, C_prev), self.num_labels),
        #     "kan_1":KanWarapper(cast(int, C_prev), self.num_labels,base_activation=nn.Hardtanh)
        # }, label='classifier1')
        # self.classifier2 = LayerChoice({
        #     "mlp_2":MutableLinear(self.num_labels*self.num_slices_per_image, self.num_labels),
        #     "kan_2":KanWarapper(self.num_labels*self.num_slices_per_image, self.num_labels, base_activation=nn.Softmax)
        # }, label='classifier2')

    def forward(self, inputs):
        # num_images, num_slices_per_image, _, height, width = inputs.size()
        # self.num_slices_per_image = num_slices_per_image
        # if self.dataset == 'imagenet':
        #     s0 = self.stem0(inputs.view(-1, 1, height, width))  # Flatten batch and channel dimensions
        #     s1 = self.stem1(s0)
        # else:
        #     s0 = s1 = self.stem(inputs.view(-1, 1, height, width))  # Flatten batch and channel dimensions
        
        if self.dataset == 'imagenet':
            s0 = self.stem0(inputs)
            s1 = self.stem1(s0)
        else:
            s0 = s1 = self.stem(inputs)

        for stage_idx, stage in enumerate(self.stages):
            if stage_idx == 2 and self.auxiliary_loss and self.training:
                assert isinstance(stage, nn.Sequential), 'Auxiliary loss is only supported for fixed architecture.'
                for block_idx, block in enumerate(stage):
                    # auxiliary loss is attached to the first cell of the last stage.
                    s0, s1 = block([s0, s1])
                    if block_idx == 0:
                        # Approach 1: treat each slice individually, not rcm to used unless the 2 3 is underfit, cant do nas
                        logits_aux = self.auxiliary_head(s1)
                        # logits_aux = self.auxiliary_head(s1, num_images, num_slices_per_image)
            else:
                s0, s1 = stage([s0, s1])

        out = self.global_pooling(s1)
        # Approach 1: treat each slice individually, not rcm to used unless the 2 3 is underfit, cant do nas
        logits = self.classifier(out.view(out.size(0), -1))
                
        # Approach 2: Put the output to get the proba of each slice, 
        # then put those proba of each slice to softmax once again to get proba of each 3d image
        # logits_per_slice = self.classifier1(out.view(out.size(0), -1))
        # logits_per_slice = logits_per_slice.view(num_images, num_slices_per_image, -1)
        # logits = self.classifier2(logits_per_slice.view(num_images, -1))

        if self.training and self.auxiliary_loss:
            return logits, logits_aux  # type: ignore
        else:
            return logits

    @classmethod
    def extra_oneshot_hooks(cls, strategy):
        from nni.nas.strategy import DARTS, RandomOneShot
        if isinstance(strategy, DARTS):
            return [NDSStageDifferentiable.mutate]
        elif isinstance(strategy, RandomOneShot):
            return [NDSStagePathSampling.mutate]
        return []

    def freeze(self, sample: Sample) -> None:
        """Freeze the model according to the sample.

        As different stages have dependencies among each other, we will recreate the whole model for simplicity.
        For weight inheritance purposes, this :meth:`freeze` might require re-writing.

        Parameters
        ----------
        sample
            The architecture dict.

        See Also
        --------
        nni.nas.nn.pytorch.MutableModule.freeze
        """
        with model_context(sample):
            return NDS(
                self.op_candidates,
                self.merge_op,
                self.num_nodes_per_cell,
                self.width,
                self.num_cells,
                self.dataset,
                self.auxiliary_loss,
                self.drop_path_prob
            )

    def set_drop_path_prob(self, drop_prob):
        """
        Set the drop probability of Drop-path in the network.
        Reference: `FractalNet: Ultra-Deep Neural Networks without Residuals <https://arxiv.org/pdf/1605.07648v4.pdf>`__.
        """
        for module in self.modules():
            if isinstance(module, DropPath):
                module.drop_prob = drop_prob

class MKNAS(NDS):
    __doc__ = """Search space proposed in
    `Progressive neural architecture search <https://arxiv.org/abs/1712.00559>`__.

    It is built upon :class:`~nni.nas.nn.pytorch.Cell`, and implemented based on :class:`~nni.nas.hub.pytorch.nasnet.NDS`.
    Its operator candidates are :attr:`~PNAS.PNAS_OPS`.
    It has 5 nodes per cell, and the output is concatenation of all nodes in the cell.
    """

    MKNAS_OPS = [
        'avg_pool_3x3',
        'max_pool_3x3',
        'skip_connect',
        'conv_3x3',
        'none',
        'sep_conv_3x3',
        'dil_conv_3x3',
        'kan_hswish',
        'kan_relu6',
        'kan_silu',
        'extra_dw',
        'invert_bottleneck',
        'conv_next',
        'ffn', 
    ]

    """The candidate operations."""

    def __init__(self,
                 width: Union[Tuple[int, ...], int] = (16, 24, 32),
                 num_cells: Union[Tuple[int, ...], int] = (4, 8, 12, 16, 20),
                 dataset: Literal['cifar', 'imagenet'] = 'cifar',
                 auxiliary_loss: bool = False,
                 drop_path_prob: float = 0.):
        super().__init__(self.MKNAS_OPS,
                         merge_op='all',
                         num_nodes_per_cell=4,
                         width=width,
                         num_cells=num_cells,
                         dataset=dataset,
                         auxiliary_loss=auxiliary_loss,
                         drop_path_prob=drop_path_prob)