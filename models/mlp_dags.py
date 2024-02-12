import math
import numpy as np
import torch
import torch.nn as nn
import warnings
from pdb import set_trace as bp

from .utils import critic_init_, Zero, get_in_out_degree, dag_str2code


class Block(nn.Module):
    def __init__(self, in_dim, out_dim, bn=False, bias=True, act='relu', preact=False):
        super(Block, self).__init__()
        self._in_dim = in_dim
        self._out_dim = out_dim
        self._bn = bn
        self._bias = bias
        self._act = act
        layer = []
        layer.append(nn.Linear(in_dim, out_dim, bias=bias))
        if bn: layer.append(nn.BatchNorm1d(out_dim))
        if self._act == 'relu':
            layer.append(nn.ReLU())
        elif self._act == 'gelu':
            layer.append(nn.GELU())
        if preact: layer = layer[::-1]
        self.layer = nn.Sequential(*layer)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if getattr(m, 'bias', None) is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                if getattr(m, 'bias', None) is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.layer(x)


# OP: 0 "zero", 1 "skip", 2 "linear"
class MLP(nn.Module):
    def __init__(self, dag, in_dim, width, act='relu', out_dim=10,
                 bn=False, bias=True,
                 preact=False,
                 ):
        super(MLP, self).__init__()
        self._dag = None
        self._in_dim = in_dim
        self._width = width
        self._act = act
        self._out_dim = out_dim
        self._bn = bn
        self._bias = bias
        self._preact = preact
        if preact:
            self._stem = nn.Linear(in_dim, width, bias=bias)
            self._readout = Block(width, out_dim, bn=bn, bias=bias, act=self._act, preact=self._preact)
        else:
            self._stem = Block(in_dim, width, bn=bn, bias=bias, act=self._act)
            self._readout = nn.Linear(width, out_dim, bias=bias)
        # _dags: _dag, _to, _from
        self._to_from_dag_layers, self._dag = self._build_dag_layers(dag)
        self._in_degree, self._out_degree = get_in_out_degree(self._dag)
        self._init()
        if preact:
            critic_init_(self._readout.layer[1].weight, mode='fan_out', nonlinearity='relu', degree=1, std_power=2)
        else:
            critic_init_(self._readout.weight, mode='fan_out', nonlinearity='relu', degree=1, std_power=2)

    def _build_dag_layers(self, dag):
        dag = dag_str2code(dag)
        _to_from_dag_layers = nn.ModuleList() # _to, _from, _dag
        for _to in range(len(dag)):
            _to_from_dag_layers.append(nn.ModuleList())
            for _from in range(len(dag[_to])):
                _to_from_dag_layers[-1].append(self._build_layer(dag[_to][_from]))
        return _to_from_dag_layers, dag

    def _build_layer(self, op):
        if op == 2:
            return Block(self._width, self._width, self._bn, bias=self._bias, act=self._act, preact=self._preact)
        elif op == 1:
            return nn.Identity()
        else:
            return Zero()

    def _init(self):
        mode = 'fan_out'
        for name, m in self.named_modules():
            if isinstance(m, nn.Linear):
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
                critic_init_(m.weight, mode=mode, nonlinearity='relu', degree=degree) # ours by degree
                # critic_init_(m.weight, mode=mode, nonlinearity='relu', degree=1) # baseline
                if getattr(m, 'bias', None) is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
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
        return _nodes, self._readout(_nodes[-1])

    def forward(self, x, return_all=False):
        x = torch.flatten(x, 1)
        x = self._stem(x)
        feature, output = self.forward_single(x)
        if return_all:
            return feature, output
        else:
            return output
