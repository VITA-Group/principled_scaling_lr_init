import numpy as np
import models


def find_all_paths(Aff, all_paths, all_paths_idx, curr_path=[], curr_path_idx=[], curr_pos=0, end_pos=5):
    if curr_pos == end_pos:
        all_paths.append(list(curr_path))
        all_paths_idx.append(list(curr_path_idx))
        return

    next_nodes = np.where(Aff[curr_pos, (curr_pos+1):] >= 0)[0] + curr_pos + 1 # -1: none; 0: skip; >=1 parameterized
    for node in next_nodes:
        curr_path.append(Aff[curr_pos, node])
        curr_path_idx.append([curr_pos, node])
        find_all_paths(Aff, all_paths, all_paths_idx, curr_path, curr_path_idx, node, end_pos)
        curr_path.pop(-1)
        curr_path_idx.pop(-1)
    return all_paths, all_paths_idx


def dag_depths(Aff):
    paths, paths_idx = find_all_paths(Aff, [], [], end_pos=len(Aff)-1)
    depths = []
    depth = 0
    for path, path_idx in zip(paths, paths_idx):
        _depth = np.sum(path)
        depth += _depth
        depths.append(_depth)
    if depth == 0:
        return [0]
    else:
        return depths


def effective_depth_width(Aff):
    paths, paths_idx = find_all_paths(Aff, [], [], end_pos=len(Aff)-1)
    width = 0
    depth = 0
    max_depth = 0 # max depth among all paths
    param_edges = [] # num. real effective parameterized edges!
    if len(paths) == 0: return 0, 0, [], 0
    for path, path_idx in zip(paths, paths_idx):
        _depth = np.sum(path)
        depth += _depth
        max_depth = max(max_depth, _depth)
        width += int(np.sum(path) > 0)
        for node, node_idx in zip(path, path_idx):
            if node == 1:
                param_edges.append("-".join([str(i) for i in node_idx]))
    depth = depth / len(paths)
    return depth, len(paths), paths, max_depth


def dag2affinity(dag):
    # e.g. "2_02_002" => [[2], [0, 2], [0, 0, 2]]
    num_nodes = len(dag) + 1
    Aff = np.ones((num_nodes, num_nodes)) * -1 # from x to
    np.fill_diagonal(Aff, 0)
    for _idx in range(len(dag)):
        to_node = _idx + 1
        edges = dag[_idx]
        for from_node, edge in enumerate(edges):
            Aff[from_node, to_node] = edge - 1 # here -1 is 0, 0 is 1, 1 is 2
    return Aff


def build_model(args, classes=10, dummy_shape=(3, 32, 32), width=None):
    if args.arch == "mlp":
        model = models.mlp(args.dag, np.prod(dummy_shape), args.width, act=args.act, out_dim=classes, bn=args.bn) #, bias=args.bias)
    elif args.arch == "cnn":
        model = models.cnn(
            args.dag, dummy_shape[0], args.width, act=args.act, out_dim=classes, kernel_size=args.kernel_size, stride=args.stride, bn=args.bn,
        )
    elif args.arch == "tinynetwork":
        model = models.__dict__[args.arch](args.width if (width is None) else width, 5, args.dag, classes)
    return model
