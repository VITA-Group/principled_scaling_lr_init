from typing import List, Text, Any
import torch
import torch.nn as nn
from .cells import InferCell
from .cell_operations import ResNetBasicblock, NAS_BENCH_201
from .genotypes import Structure as CellStructure
from .utils import _init, MODE, get_in_out_degree, dag_str2code

from pdb import set_trace as bp


def code2arch_str(code):
    # 3_34_131
    # '|nor_conv_3x3~0|+|nor_conv_3x3~0|avg_pool_3x3~1|+|skip_connect~0|nor_conv_3x3~1|skip_connect~2|'
    nodes = []
    for code_node in code.split('_'):
        _node = []
        for index, edge in enumerate(code_node):
            _node.append(NAS_BENCH_201[int(edge)] + "~" + str(index))
        nodes.append("|" + "|".join(_node) + "|")
    return "+".join(nodes)


# https://github.com/D-X-Y/AutoDL-Projects/blob/main/exps/basic/basic-main.py
# The macro structure for architectures in NAS-Bench-201
class TinyNetwork(nn.Module):

    def __init__(self, C, N, dag, num_classes, mup=False):
        super(TinyNetwork, self).__init__()
        self._C = C
        self._layerN = N
        self.dag = dag_str2code(dag)
        self.in_degree, self.out_degree = get_in_out_degree(self.dag)

        self.genotype = CellStructure.str2structure(code2arch_str(dag))
        self.mup = mup

        self.stem = nn.Sequential(
            nn.Conv2d(3, C, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(C)
        )

        layer_channels = [C] * N + [C*2] + [C*2] * N + [C*4] + [C*4] * N
        layer_reductions = [False] * N + [True] + [False] * N + [True] + [False] * N
        C_prev = C
        _init(self.stem, degree=1, mode=MODE)

        self.cells = nn.ModuleList()
        for index, (C_curr, reduction) in enumerate(zip(layer_channels, layer_reductions)):
            if reduction:
                cell = ResNetBasicblock(C_prev, C_curr, 2, True)
            else:
                cell = InferCell(self.genotype, C_prev, C_curr, 1, degrees=[self.in_degree, self.out_degree]) # by degree
            self.cells.append(cell)
            C_prev = cell.out_dim
        self._Layer = len(self.cells)

        self.lastact = nn.Sequential(nn.BatchNorm2d(C_prev), nn.ReLU(inplace=True))
        self.global_pooling = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Linear(C_prev, num_classes)

        _init(self.lastact, degree=1, mode=MODE)
        _init(self.classifier, degree=1, mode=MODE, std_power=2)

    def get_message(self):
        string = self.extra_repr()
        for i, cell in enumerate(self.cells):
            string += '\n {:02d}/{:02d} :: {:}'.format(
                i, len(self.cells), cell.extra_repr())
        return string

    def extra_repr(self):
        return ('{name}(C={_C}, N={_layerN}, L={_Layer})'.format(name=self.__class__.__name__, **self.__dict__))

    def forward(self, inputs):
        feature = self.stem(inputs)
        for i, cell in enumerate(self.cells):
            feature = cell(feature)

        out = self.lastact(feature)
        out = self.global_pooling(out)
        out = out.view(out.size(0), -1)
        logits = self.classifier(out)

        return logits
