import warnings
import math
import numpy as np
import torch
from torch import nn



MODE = 'fan_in'


def dag_str2code(dag):
    # list of to_node (list of in_node). 0: broken; 1: skip-connect; 2: linear or conv
    # e.g. "2_02_002" => [[2], [0, 2], [0, 0, 2]]
    if isinstance(dag, str):
        dag = [[int(edge) for edge in node] for node in dag.split('_')]
    elif isinstance(dag, list):
        assert isinstance(dag[0], list) and len(dag[0]) == 1 # 2nd node has one in-degree
        for i in range(1, len(dag)):
            assert len(dag[i]) == len(dag[i-1]) + 1 # next node has one more in-degree than prev node
    return dag


# DAG: 0 - zero, 1 - skip, 2 - linear-relu
# 201: 0 - zero, 1 - skip, 2 - conv1x1, 3 - conv3x3, 4 - avg_pool_3x3
def get_in_out_degree(dag):
    in_degree = [1]
    out_degree = []
    for node in dag:
        in_degree.append(sum(np.array(node) > 0))
    in_degree += [1] # read_out
    for node_idx in range(len(dag[-1])):
        out_degree.append(sum(np.array([node[node_idx] if len(node) > node_idx else 0 for node in dag]) > 0))
    out_degree += [1, 1] # final dag layer & read_out
    return in_degree, out_degree


# https://github.com/pytorch/pytorch/blob/master/torch/nn/init.py#L415
def critic_init_(
    tensor: torch.Tensor, a: float = 0, mode: str = 'fan_in', nonlinearity: str = 'leaky_relu', degree: int = 1,
    std_power: int = 1
):
    r"""Fills the input `Tensor` with values according to the method
    described in `Delving deep into rectifiers: Surpassing human-level
    performance on ImageNet classification` - He, K. et al. (2015), using a
    normal distribution. The resulting tensor will have values sampled from
    :math:`\mathcal{N}(0, \text{std}^2)` where
    .. math::
        \text{std} = \frac{\text{gain}}{\sqrt{\text{fan\_mode}}}
    Also known as He initialization.
    Args:
        tensor: an n-dimensional `torch.Tensor`
        a: the negative slope of the rectifier used after this layer (only
            used with ``'leaky_relu'``)
        mode: either ``'fan_in'`` (default) or ``'fan_out'``. Choosing ``'fan_in'``
            preserves the magnitude of the variance of the weights in the
            forward pass. Choosing ``'fan_out'`` preserves the magnitudes in the
            backwards pass.
        nonlinearity: the non-linear function (`nn.functional` name),
            recommended to use only with ``'relu'`` or ``'leaky_relu'`` (default).
    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.kaiming_normal_(w, mode='fan_out', nonlinearity='relu')
    """
    if 0 in tensor.shape:
        warnings.warn("Initializing zero-element tensors is a no-op")
        return tensor
    # https://github.com/pytorch/pytorch/blob/master/torch/nn/init.py#L284
    fan = torch.nn.init._calculate_correct_fan(tensor, mode)
    gain = torch.nn.init.calculate_gain(nonlinearity, a)
    gain /= math.sqrt(degree) # TODO
    std = gain / math.sqrt(fan)
    with torch.no_grad():
        return tensor.normal_(0, std**std_power)


def _init(model, degree=1, mode='fan_out', std_power=1):
    degree = max(degree, 1)
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            # nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            critic_init_(m.weight, mode=mode, nonlinearity='relu', degree=degree, std_power=std_power)
            # critic_init_(m.weight, mode=mode, nonlinearity='relu', std_power=std_power)
            if getattr(m, 'bias', None) is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            if getattr(m, 'bias', None) is not None:
                nn.init.constant_(m.bias, 0)


class Zero(nn.Module):
    def __init__(self):
        super(Zero, self).__init__()

    def forward(self, x):
        return x * 0
