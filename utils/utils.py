import os
import os.path as osp
import random
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.utils import (to_scipy_sparse_matrix, scatter, )
import inspect
from typing import Any, Dict, List, Optional, Union
import time
from datetime import datetime
from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint

from torch import Tensor
from torchmetrics import AUROC, AveragePrecision
import yaml



class MLP(torch.nn.Module):
    """
    MLP model modifed from pytorch geometric.
    """

    def __init__(
        self,
        channel_list: Optional[Union[List[int], int]] = None,
        dropout: Union[float, List[float]] = 0.0,
        act: Union[str, Callable, None] = "relu",
        act_first: bool = False,
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm: bool = True,
        plain_last: bool = True,
        bias: Union[bool, List[bool]] = True,
        **kwargs,
    ):
        super().__init__()

        assert isinstance(channel_list, (tuple, list))
        assert len(channel_list) >= 2
        self.channel_list = channel_list

        self.act = activation_resolver(act, **(act_kwargs or {}))
        self.act_first = act_first
        self.plain_last = plain_last

        if isinstance(dropout, float):
            dropout = [dropout] * (len(channel_list) - 1)
            if plain_last:
                dropout[-1] = 0.0
        elif len(dropout) != len(channel_list) - 1:
            raise ValueError(
                f"Number of dropout values provided ({len(dropout)} does not "
                f"match the number of layers specified "
                f"({len(channel_list)-1})"
            )
        self.dropout = dropout

        if isinstance(bias, bool):
            bias = [bias] * (len(channel_list) - 1)
        if len(bias) != len(channel_list) - 1:
            raise ValueError(
                f"Number of bias values provided ({len(bias)}) does not match "
                f"the number of layers specified ({len(channel_list)-1})"
            )

        self.lins = torch.nn.ModuleList()
        iterator = zip(channel_list[:-1], channel_list[1:], bias)
        for in_channels, out_channels, _bias in iterator:
            self.lins.append(
                torch.nn.Linear(in_channels, out_channels, bias=_bias)
            )

        self.norms = torch.nn.ModuleList()
        iterator = channel_list[1:-1] if plain_last else channel_list[1:]
        for hidden_channels in iterator:
            if norm is not None:
                norm_layer = torch.nn.BatchNorm1d(hidden_channels)
            else:
                norm_layer = torch.nn.Identity()
            self.norms.append(norm_layer)

        self.reset_parameters()

    @property
    def in_channels(self) -> int:
        r"""Size of each input sample."""
        return self.channel_list[0]

    @property
    def out_channels(self) -> int:
        r"""Size of each output sample."""
        return self.channel_list[-1]

    @property
    def num_layers(self) -> int:
        r"""The number of layers."""
        return len(self.channel_list) - 1

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        for lin in self.lins:
            lin.reset_parameters()
        for norm in self.norms:
            if hasattr(norm, "reset_parameters"):
                norm.reset_parameters()

    def forward(self, x: Tensor) -> Tensor:
        r"""
        Args:
            x (torch.Tensor): The source tensor.
            return_emb (bool, optional): If set to :obj:`True`, will
                additionally return the embeddings before execution of to the
                final output layer. (default: :obj:`False`)
        """
        for i, (lin, norm) in enumerate(zip(self.lins, self.norms)):
            x = lin(x)
            if self.act is not None and self.act_first:
                x = self.act(x)
            x = norm(x)
            if self.act is not None and not self.act_first:
                x = self.act(x)
            x = F.dropout(x, p=self.dropout[i], training=self.training)

        if self.plain_last:
            x = self.lins[-1](x)
            x = F.dropout(x, p=self.dropout[-1], training=self.training)

        return x

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({str(self.channel_list)[1:-1]})"

def swish(x: Tensor) -> Tensor:
    return x * x.sigmoid()

def normalize_string(s: str) -> str:
    return s.lower().replace("-", "").replace("_", "").replace(" ", "")

def resolver(
    classes: List[Any],
    class_dict: Dict[str, Any],
    query: Union[Any, str],
    base_cls: Optional[Any],
    base_cls_repr: Optional[str],
    *args,
    **kwargs,
):

    if not isinstance(query, str):
        return query

    query_repr = normalize_string(query)
    if base_cls_repr is None:
        base_cls_repr = base_cls.__name__ if base_cls else ""
    base_cls_repr = normalize_string(base_cls_repr)

    for key_repr, cls in class_dict.items():
        if query_repr == key_repr:
            if inspect.isclass(cls):
                obj = cls(*args, **kwargs)
                assert callable(obj)
                return obj
            assert callable(cls)
            return cls

    for cls in classes:
        cls_repr = normalize_string(cls.__name__)
        if query_repr in [cls_repr, cls_repr.replace(base_cls_repr, "")]:
            if inspect.isclass(cls):
                obj = cls(*args, **kwargs)
                assert callable(obj)
                return obj
            assert callable(cls)
            return cls

    choices = set(cls.__name__ for cls in classes) | set(class_dict.keys())
    raise ValueError(f"Could not resolve '{query}' among choices {choices}")


# Activation Resolver #########################################################

def activation_resolver(query: Union[Any, str] = "relu", *args, **kwargs):
    import torch

    base_cls = torch.nn.Module
    base_cls_repr = "Act"
    acts = [
        act
        for act in vars(torch.nn.modules.activation).values()
        if isinstance(act, type) and issubclass(act, base_cls)
    ]
    acts += [
        swish,
    ]
    act_dict = {}
    return resolver(
        acts, act_dict, query, base_cls, base_cls_repr, *args, **kwargs
    )



class SmartTimer:
    """A timer utility that output the elapsed time between this
    call and last call.
    """

    def __init__(self, verb=True) -> None:
        """SmartTimer Constructor

        Keyword Arguments:
            verb {bool} -- Controls printing of the timer (default: {True})
        """
        self.last = time.time()
        self.verb = verb

    def record(self):
        """Record current timestamp"""
        self.last = time.time()

    def cal_and_update(self, name):
        """Record current timestamp and print out time elapsed from last
        recorded time.

        Arguments:
            name {string} -- identifier of the printout.
        """
        now = time.time()
        if self.verb:
            print(name, now - self.last)
        self.record()

def get_label_texts(labels):
    label_texts = [None] * int(len(labels) * 2)
    for entry in labels:
        label_texts[labels[entry][0]] = (
                "prompt node. molecule property description. " + "The molecule is effective to the following assay. " +
                labels[entry][1][0][:-41])
        label_texts[labels[entry][0] + len(labels)] = (
                "prompt node. molecule property description. " + "The molecule is not effective to the following "
                                                                 "assay. " +
                labels[entry][1][0][:-41])
    return label_texts


def dict_res_summary(res_col):
    """Combine multiple dictionary information into one dictionary
    so that all entries with the same key will be concatenated into
    a list

    Arguments:
        res_col {list[dictionary]} -- a list of dictionary

    Returns:
        dictionary -- summarized dictionary information
    """
    res_dict = {}
    for res in res_col:
        for k in res:
            if k not in res_dict:
                res_dict[k] = []
            res_dict[k].append(res[k])
    return res_dict

def load_pretrained_state(model_dir, deepspeed=False):
    if deepspeed:
        def _remove_prefix(key: str, prefix: str) -> str:
            return key[len(prefix):] if key.startswith(prefix) else key
        state_dict = get_fp32_state_dict_from_zero_checkpoint(model_dir)
        state_dict = {_remove_prefix(k, "_forward_module."): state_dict[k] for k in state_dict}
    else:
        state_dict = torch.load(model_dir)["state_dict"]
    return state_dict

def combine_dict(*args):
    combined_dict = {}
    for d in args:
        for k in d:
            combined_dict[k] = d[k]
    return combined_dict

def merge_mod(params, mod_args):
    for i in range(0, len(mod_args), 2):
        if mod_args[i + 1].isdigit():
            val = int(mod_args[i + 1])
        elif mod_args[i + 1].replace(".", "", 1).isdigit():
            val = float(mod_args[i + 1])
        elif mod_args[i + 1].lower() == "true":
            val = True
        elif mod_args[i + 1].lower() == "false":
            val = False
        else:
            val = mod_args[i + 1]
        params[mod_args[i]] = val
    return params

def setup_exp(params):
    if not osp.exists("./saved_exp"):
        os.mkdir("./saved_exp")

    curtime = datetime.now()
    exp_dir = osp.join("./saved_exp", str(curtime))
    os.mkdir(exp_dir)
    with open(osp.join(exp_dir, "command"), "w") as f:
        yaml.dump(params, f)
    params["exp_dir"] = exp_dir

def set_random_seed(seed):
    """Set python, numpy, pytorch global random seed.
    Does not guarantee determinism due to PyTorch's feature.

    Arguments:
        seed {int} -- Random seed to set
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

class MultiApr(torch.nn.Module):
    def __init__(self, num_labels=1):
        super().__init__()
        self.metrics = torch.nn.ModuleList([AveragePrecision(task="binary") for i in range(num_labels)])

    def update(self, preds, targets):
        for i, met in enumerate(self.metrics):
            pred = preds[:, i]
            target = targets[:, i]
            valid_idx = target == target
            # print(pred[valid_idx])
            # print(target[valid_idx])
            met.update(pred[valid_idx], target[valid_idx].to(torch.long))

    def compute(self):
        full_val = []
        for met in self.metrics:
            try:
                res = met.compute()
                if res == res:
                    full_val.append(res)
            except BaseException:
                pass
        return torch.tensor(full_val).mean()

    def reset(self):
        for met in self.metrics:
            met.reset()


class MultiAuc(torch.nn.Module):
    def __init__(self, num_labels=1):
        super().__init__()
        self.metrics = torch.nn.ModuleList([AUROC(task="binary") for i in range(num_labels)])

    def update(self, preds, targets):
        for i, met in enumerate(self.metrics):
            pred = preds[:, i]
            target = targets[:, i]
            valid_idx = target == target
            # print(pred[valid_idx])
            # print(target[valid_idx])
            met.update(pred[valid_idx], target[valid_idx].to(torch.long))

    def compute(self):
        full_val = []
        for met in self.metrics:
            try:
                res = met.compute()
                if res == res:
                    full_val.append(res)
            except BaseException:
                pass
        return torch.tensor(full_val).mean()

    def reset(self):
        for met in self.metrics:
            met.reset()

def scipy_rwpe(data, walk_length):
    row, col = data.edge_index
    N = data.num_nodes

    value = data.edge_weight
    if value is None:
        value = torch.ones(data.num_edges, device=row.device)
    value = scatter(value, row, dim_size=N, reduce="sum").clamp(min=1)[row]
    value = 1.0 / value
    adj = to_scipy_sparse_matrix(data.edge_index, edge_attr=value, num_nodes=data.num_nodes)

    out = adj
    pe_list = [out.diagonal()]
    for _ in range(walk_length - 1):
        out = out @ adj
        pe_list.append(out.diagonal())
    pe = torch.tensor(np.stack(pe_list, axis=-1))

    return pe

def set_mask(data, name, index, dtype=torch.bool):
    mask = torch.zeros(data.num_nodes, dtype=dtype)
    mask[index] = True
    setattr(data, name, mask)

def load_yaml(dir):
    with open(dir, "r") as stream:
        return yaml.safe_load(stream)
    
def get_available_devices():
    r"""Get IDs of all available GPUs.

    Returns:
        device (torch.device): Main device (GPU 0 or CPU).
        gpu_ids (list): List of IDs of all GPUs that are available.
    """
    gpu_ids = []
    if torch.cuda.is_available():
        gpu_ids += [gpu_id for gpu_id in range(torch.cuda.device_count())]
        device = torch.device(f'cuda:{gpu_ids[0]}')
        torch.cuda.set_device(device)
    else:
        device = torch.device('cpu')

    return device, gpu_ids


def binary_apr_func(func, output, batch):
    output = output.view(-1, batch.num_classes[0])
    score = torch.sigmoid(output)
    label = batch.bin_labels[batch.true_nodes_mask]
    return func.update(score, label.view(len(batch), -1))


def binary_auc_multi_func(func, output, batch):
    output = output.view(-1, batch.num_classes[0])
    score = torch.sigmoid(output)
    label = batch.bin_labels[batch.true_nodes_mask]
    return func.update(score, label.view(-1, batch.num_classes[0]))


def binary_single_auc_func(func, output, batch):
    output = output.view(-1, batch.num_classes[0])
    score = torch.sigmoid(output)
    # if len(score.unique()) == 1:
    # print(output[:20])
    label = batch.bin_labels[batch.true_nodes_mask]
    # print(score)
    # print(label)
    return func.update(score, label.view(-1, batch.num_classes[0]))


def classification_single_func(func, output, batch):
    label = batch.bin_labels[batch.true_nodes_mask].view(-1, batch.num_classes[0])
    output = output.view(-1, batch.num_classes[0])
    return func(output, torch.argmax(label, dim=-1))

def flat_auc(func, output, batch):
    return func(torch.sigmoid(output).view(-1), batch.bin_labels[batch.true_nodes_mask].view(-1))
