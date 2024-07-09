from functools import partial
from typing import Tuple, List, Union, Iterable, Literal, Dict, Callable, Optional, cast, Type, Iterator

from torch import nn

import nni
from nni.mutable import MutableExpression, Sample
from nni.nas.nn.pytorch import ModelSpace, LayerChoice, Cell, MutableConv2d, MutableBatchNorm2d, MutableLinear, model_context
from nni.nas.hub.pytorch.utils.nn import DropPath
from nni.nas.hub.pytorch.nasnet import (
    Zero, FactorizedReduce, CellPreprocessor, CellPostprocessor, NDSStage, NDSStageDifferentiable, NDSStagePathSampling
)

from Model.architecture.KANLinear import KanWarapper
from Model.architecture.MobileVitV4 import UniversialInvertedResidual, _se_or_skip

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
    'conv_3x3': lambda C, stride, affine:
        nn.Sequential(
            nn.ReLU(inplace=False),
            MutableConv2d(C, C, 3, stride=stride, padding=1, bias=False),
            MutableBatchNorm2d(C, affine=affine)
        ),
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


class AuxiliaryHead(nn.Module):
    def __init__(self, C: int, num_labels: int):
        super().__init__()
        stride = 2

        self.features = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.AvgPool2d(5, stride=stride, padding=0, count_include_pad=False),
            MutableConv2d(C, 32, 1, bias=False),
            MutableBatchNorm2d(32),
            nn.ReLU(inplace=True),
            MutableConv2d(32, 128, 2, bias=False),
            MutableBatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.classifier1 = MutableLinear(128, num_labels)
        self.classifier2 = MutableLinear(num_labels*6, num_labels)

    def forward(self, x, num_images, num_slices_per_image):
            x = self.features(x)
            x = self.classifier1(x.view(x.size(0), -1))
            x = x.view(num_images, num_slices_per_image, -1)
            x = self.classifier2(x.view(num_images, -1))
            return x


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

class MobileKAN(ModelSpace):
    __doc__ = """
    The implementation of NASNet search space. Here I modfify the NDS search space of NNI library

    The NDS search space follows the implementation in
    `unnas <https://github.com/facebookresearch/unnas/blob/main/pycls/models/nas/nas.py>`__.
    See `On Network Design Spaces for Visual Recognition <https://arxiv.org/abs/1905.13214>`__ for details.

    This is the model space design to classify 3d images by performing operation on 2d slices of those images
    The recommend number of slice is 6 slices per image, with original shape of each 2d slices is 224x224
    
    Parameters
    ----------
    width
        A fixed initial width or a tuple of widths to choose from.
    num_cells
        A fixed number of cells (depths) to stack, or a tuple of depths to choose from.
    auxiliary_loss
        If true, another auxiliary classification head will produce the another prediction.
        This makes the output of network two logits in the training phase.
    drop_path_prob
        Apply drop path. Enabled when it's set to be greater than 0.
    num_classes
        Number of class to classify
    num_slices_per_image
        Number of 2d slices taken from 3d image
    num_nodes_per_cell
        Number of nodes in the cell
    """

    def __init__(self,
                 width: Union[Tuple[int, ...], int] = 16,
                 num_cells: Union[Tuple[int, ...], int] = 20,
                 auxiliary_loss: bool = False,
                 drop_path_prob: float = 0.,
                 num_classes: int = 3,
                 num_slices_per_image: int = 6,
                 num_nodes_per_cell: int = 4):
        super().__init__()

        self.op_candidates = [
            'avg_pool_3x3',
            'max_pool_3x3',
            'skip_connect',
            'conv_3x3',
            'none',
            'kan_hswish',
            'kan_relu6',
            'kan_silu',
            'extra_dw',
            'invert_bottleneck',
            'conv_next',
            'ffn', 
        ]
        self.merge_op = 'all'
        self.width = width
        self.num_cells = num_cells
        self.num_classes = num_classes
        self.auxiliary_loss = auxiliary_loss
        self.drop_path_prob = drop_path_prob
        self.num_slices_per_image = num_slices_per_image
        self.num_nodes_per_cell = num_nodes_per_cell

        # preprocess the specified width and depth
        if isinstance(width, Iterable):
            C = nni.choice('width', list(width))
        else:
            C = width

        self.num_cells: MaybeIntChoice = cast(int, num_cells)
        if isinstance(num_cells, Iterable):
            self.num_cells = nni.choice('depth', list(num_cells))
        num_cells_per_stage = [(i + 1) * self.num_cells // 3 - i * self.num_cells // 3 for i in range(3)]

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

        self.stages = nn.ModuleList()
        for stage_idx in range(3):
            if stage_idx > 0:
                C_curr *= 2
            # For a stage, we get C_in, C_curr, and C_out.
            # C_in is only used in the first cell.
            # C_curr is number of channels for each operator in current stage.
            # C_out is usually `C * num_nodes_per_cell` because of concat operator.
            cell_builder = CellBuilder(self.op_candidates, C_pprev, C_prev, C_curr, self.num_nodes_per_cell,
                                       self.merge_op, stage_idx > 0, last_cell_reduce, drop_path_prob)
            stage: Union[NDSStage, nn.Sequential] = NDSStage(cell_builder, num_cells_per_stage[stage_idx])

            if isinstance(stage, NDSStage):
                stage.estimated_out_channels_prev = cast(int, C_prev)
                stage.estimated_out_channels = cast(int, C_curr * self.num_nodes_per_cell)
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
            self.auxiliary_head = AuxiliaryHead(C_to_auxiliary, self.num_classes)  # type: ignore


        self.global_pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier1 = LayerChoice({
            "mlp_1":MutableLinear(cast(int, C_prev), self.num_classes),
            "kan_1":KanWarapper(cast(int, C_prev), self.num_classes,base_activation=nn.Softmax)
        }, label='classifier1')
        self.classifier2 = LayerChoice({
            "mlp_2":MutableLinear(self.num_classes*self.num_slices_per_image, self.num_classes),
            "kan_2":KanWarapper(self.num_classes*self.num_slices_per_image, self.num_classes, base_activation=nn.Softmax)
        }, label='classifier2')

    def forward(self, inputs):
        num_images, num_slices_per_image, _, height, width = inputs.size()
        self.num_slices_per_image = num_slices_per_image
        s0 = self.stem0(inputs.view(-1, 1, height, width))
        s1 = self.stem1(s0)

        for stage_idx, stage in enumerate(self.stages):
            if stage_idx == 2 and self.auxiliary_loss and self.training:
                assert isinstance(stage, nn.Sequential), 'Auxiliary loss is only supported for fixed architecture.'
                for block_idx, block in enumerate(stage):
                    # auxiliary loss is attached to the first cell of the last stage.
                    s0, s1 = block([s0, s1])
                    if block_idx == 0:
                        logits_aux = self.auxiliary_head(s1, num_images, num_slices_per_image)
            else:
                s0, s1 = stage([s0, s1])

        out = self.global_pooling(s1)
        logits_per_slice = self.classifier1(out.view(out.size(0), -1))
        logits_per_slice = logits_per_slice.view(num_images, num_slices_per_image, -1)
        logits = self.classifier2(logits_per_slice.view(num_images, -1))

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
            return MobileKAN(
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
                module.drop_prob = drop_prob