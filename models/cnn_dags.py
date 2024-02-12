import math
import numpy as np
import torch
import torch.nn as nn
import warnings
from pdb import set_trace as bp

from .utils import critic_init_, get_in_out_degree, Zero


class Block(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size=3, stride=1, bn=False, bias=True, act='relu'):
        super(Block, self).__init__()
        self._in_dim = in_dim
        self._out_dim = out_dim
        self._kernel_size = kernel_size
        self._stride = stride
        self._padding = (1 * (kernel_size - 1) + 1 - stride) // 2
        self._bn = bn
        self._bias = bias
        self._act = act
        layer = []
        layer.append(nn.Conv2d(in_dim, out_dim, kernel_size, stride=stride, padding=self._padding, bias=bias))
        if bn: layer.append(nn.BatchNorm2d(out_dim))
        if self._act == 'relu':
            layer.append(nn.ReLU(inplace=True))
        elif self._act == 'gelu':
            layer.append(nn.GELU())
        self.layer = nn.Sequential(*layer)

    def forward(self, x):
        return self.layer(x)


# OP: 0 "zero", 1 "skip", 2 "linear"
class CNN(nn.Module):
    def __init__(self, dag, in_dim, width, act='relu', out_dim=10, kernel_size=3, stride=1,
                 bn=False, bias=True):
        super(CNN, self).__init__()
        self._dag = None
        self._in_dim = in_dim
        self._width = width
        self._act = act
        self._act = act
        self._out_dim = out_dim
        self._kernel_size = kernel_size
        self._stride = stride
        self._bn = bn
        self._bias = bias
        self._stem = Block(in_dim, width, kernel_size=kernel_size, stride=stride, bn=bn, bias=bias, act=self._act)
        self._readout = nn.Linear(width, out_dim, bias=bias)
        # _dags: _dag, _to, _from
        self._to_from_dag_layers, self._dag = self._build_dag_layers(dag)
        self._in_degree, self._out_degree = get_in_out_degree(self._dag)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self._init()
        critic_init_(self._readout.weight, mode='fan_out', nonlinearity='relu', degree=1, std_power=2)

    def _build_dag_layers(self, dag):
        # list of to_node (list of in_node). 0: broken; 1: skip-connect; 2: linear or conv
        # e.g. "2_02_002" => [[2], [0, 2], [0, 0, 2]]
        if isinstance(dag, str):
            dag = [[int(edge) for edge in node] for node in dag.split('_')]
        elif isinstance(dag, list):
            assert isinstance(dag[0], list) and len(dag[0]) == 1 # 2nd node has one in-degree
            for i in range(1, len(dag)):
                assert len(dag[i]) == len(dag[i-1]) + 1 # next node has one more in-degree than prev node
        _to_from_dag_layers = nn.ModuleList() # _to, _from, _dag
        for _to in range(len(dag)):
            _to_from_dag_layers.append(nn.ModuleList())
            for _from in range(len(dag[_to])):
                _to_from_dag_layers[-1].append(self._build_layer(dag[_to][_from]))
        return _to_from_dag_layers, dag

    def _build_layer(self, op):
        if op == 2:
            return Block(self._width, self._width, kernel_size=self._kernel_size, stride=self._stride, bn=self._bn, bias=self._bias, act=self._act)
        elif op == 1:
            return nn.Identity()
        else:
            return Zero()

    def _init(self):
        mode = 'fan_out'
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                if '_to_from_dag' in name:
                    _to, _from = [int(value) for value in name[name.index('layer')+7: name.index('layer.')-1].split('.')]
                    _to += 1
                inD = outD = 1
                if 'stem' in name:
                    inD = self._in_degree[0]
                    outD = self._out_degree[0]
                elif 'readout' in name:
                    inD = self._in_degree[-1]
                    outD = self._out_degree[-1]
                else:
                    inD = self._in_degree[_from]
                    outD = self._out_degree[_from]
                degree = max(inD, 1)
                critic_init_(m.weight, mode=mode, nonlinearity='relu', degree=degree)
                # critic_init_(m.weight, mode=mode, nonlinearity='relu')
                if getattr(m, 'bias', None) is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                if getattr(m, 'bias', None) is not None:
                    nn.init.constant_(m.bias, 0)

    def _get_dag_layers(self, dag_idx):
        layers = []
        for _to in range(len(self._dag)):
            for _from in range(len(self._dag[_to])):
                layers.append(self._to_from_dag_layers[_to][_from][dag_idx])
        return layers

    def forward_single(self, x):
        _nodes = [x] # output from prev node, input to next node
        for _to in range(len(self._dag)):
            _node = []
            for _from in range(len(self._dag[_to])):
                _node.append(self._to_from_dag_layers[_to][_from](_nodes[_from]))
            _nodes.append(sum(_node))
        return _nodes

    def forward(self, x, return_all=False):
        x = self._stem(x)
        feature = self.forward_single(x)
        output = self.avgpool(feature[-1])
        output = torch.flatten(output, 1)
        output = self._readout(output)
        if return_all:
            return feature, output
        else:
            return output
