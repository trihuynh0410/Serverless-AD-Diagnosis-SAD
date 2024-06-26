from functools import partial
from typing import Tuple, List, Union, Iterable, Literal, Dict, Callable, Optional, cast

import torch
from torch import nn

import nni
from nni.mutable import MutableExpression, Sample
from nni.nas.oneshot.pytorch.supermodule.sampling import PathSamplingRepeat
from nni.nas.oneshot.pytorch.supermodule.differentiable import DifferentiableMixedRepeat
from nni.nas.nn.pytorch import ModelSpace, Repeat, Cell, MutableConv2d, MutableBatchNorm2d, model_context,LayerChoice, MutableLinear
from nni.nas.hub.pytorch.utils.nn import DropPath

from KANLinear import KanWarapper
from MobileNetV4 import _se_or_skip, ConvBNReLU, UniversialInvertedResidual
from ViT import *
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
            self.pre0 = ConvBNReLU(cast(int, C_pprev), cast(int, C), 1, 1)
        self.pre1 = ConvBNReLU(cast(int, C_prev), cast(int, C), 1, 1)

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
            return ConvBNReLU(self.estimated_out_channels_prev, self.estimated_out_channels, 1, 1, 0)
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
                 num_slices_per_image: int = 6,
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
        self.num_slices_per_image = num_slices_per_image
        self.patches_num = None
        self.C_prev = None
        self.cls_token = None
        self.pos_embed = None
        self.transformer = None
        self.norm = None
        self.classifier = None
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
            self.stem = nn.Sequential(
                MutableConv2d(3, cast(int, 3 * C), 3, padding=1, bias=False),
                MutableBatchNorm2d(cast(int, 3 * C))
            )
            C_pprev = C_prev = 3 * C
            C_curr = C
            last_cell_reduce = False

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
        
        self.global_pooling = nn.AdaptiveAvgPool2d((1, 1))

    def update_dynamic_layers(self, new_patches_num, new_C_prev):
        if self.patches_num != new_patches_num or self.C_prev != new_C_prev:
            self.patches_num = new_patches_num
            self.C_prev = new_C_prev
            self.cls_token = ClassToken(self.C_prev)
            self.pos_embed = AbsolutePositionEmbedding(self.patches_num + 1, self.C_prev)
            self.transformer = TransformerEncoderLayer(embed_dim=self.C_prev,num_heads=4,mlp_ratio=3,act_fn=torch.nn.Hardswish)
            self.norm = MutableLayerNorm(self.C_prev)
            # self.classifier = MutableLinear(self.C_prev, self.num_labels)
            self.classifier = KanWarapper(self.C_prev, self.num_labels, base_activation=nn.Softmax)

    def forward(self, inputs):
        num_images, num_slices_per_image, _, height, width = inputs.size()
        self.num_slices_per_image = num_slices_per_image
        if self.dataset == 'imagenet':
            s0 = s1 = self.stem(inputs)
        for stage_idx, stage in enumerate(self.stages):
            s0, s1 = stage([s0, s1])
        out = self.global_pooling(s1)
        new_C_prev = out.size(1)*out.size(2)*out.size(3)

        self.update_dynamic_layers(num_slices_per_image, new_C_prev)
        out = out.view(num_images, num_slices_per_image, new_C_prev)
        
        x = self.cls_token(out)
        x = self.pos_embed(x)
        x = self.transformer(x)
        self.norm.to(x.device)
        x = self.norm(x)
        x = torch.mean(x[:, 1:], dim=1)
        self.classifier = self.classifier.to(x.device)
        logits = self.classifier(x)

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
                self.num_slices_per_image,
                self.drop_path_prob
            )

    def set_drop_path_prob(self, drop_prob):
        """
        Set the drop probability of Drop-path in the network.
        Reference: `FractalNet: Ultra-Deep Neural Networks without Residuals <https://arxiv.org/pdf/1605.07648v4.pdf>`__.
        """
        for module in self.modules():
            if isinstance(module, DropPath):
                module.drop_prob = self.drop_path_prob

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
        'none',
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
                 num_slices_per_image: int = 6,
                 drop_path_prob: float = 0.):
        super().__init__(self.MKNAS_OPS,
                         merge_op='all',
                         num_nodes_per_cell=4,
                         width=width,
                         num_cells=num_cells,
                         dataset=dataset,
                         auxiliary_loss=auxiliary_loss,
                         num_slices_per_image = num_slices_per_image,
                         drop_path_prob=drop_path_prob)