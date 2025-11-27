"""Base message-passing GNNs
"""
from abc import ABCMeta, abstractmethod
from typing import Callable, Optional

import torch
import torch.nn as nn
from torch.nn import Linear, Sequential, ReLU
import torch.nn.functional as F
from torch import Tensor 
from torch_scatter import scatter
from torch_geometric.nn import MessagePassing
from torch.nn import Parameter
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.typing import Adj, OptTensor
from torch_geometric.utils import softmax, add_self_loops, degree

from utils.utils import MLP, SmartTimer


class MultiLayerMessagePassing(nn.Module, metaclass=ABCMeta):
    """Message passing GNN"""

    def __init__(
            self,
            num_layers,
            inp_dim,
            out_dim,
            drop_ratio=None,
            JK="last",
            batch_norm=True,
    ):
        """

        :param num_layers: layer number of GNN
        :type num_layers: int
        :param inp_dim: input feature dimension
        :type inp_dim: int
        :param out_dim: output dimension
        :type out_dim: int
        :param drop_ratio: layer-wise node dropout ratio, defaults to None
        :type drop_ratio: float, optional
        :param JK: jumping knowledge, should either be ["last","sum"],
        defaults to "last"
        :type JK: str, optional
        :param batch_norm: Use node embedding batch normalization, defaults
        to True
        :type batch_norm: bool, optional
        """
        super().__init__()
        self.num_layers = num_layers
        self.JK = JK
        self.inp_dim = inp_dim
        self.out_dim = out_dim

        self.drop_ratio = drop_ratio

        self.conv = torch.nn.ModuleList()

        if batch_norm:
            self.batch_norm = torch.nn.ModuleList()
            for layer in range(num_layers):
                self.batch_norm.append(torch.nn.BatchNorm1d(out_dim))
        else:
            self.batch_norm = None

        self.timer = SmartTimer(False)

    def build_layers(self):
        for layer in range(self.num_layers):
            if layer == 0:
                self.conv.append(self.build_input_layer())
            else:
                self.conv.append(self.build_hidden_layer())

    @abstractmethod
    def build_input_layer(self):
        pass

    @abstractmethod
    def build_hidden_layer(self):
        pass

    @abstractmethod
    def layer_forward(self, layer, message):
        pass

    @abstractmethod
    def build_message_from_input(self, g):
        pass

    @abstractmethod
    def build_message_from_output(self, g, output):
        pass

    def forward(self, g, drop_mask=None):
        h_list = []

        message = self.build_message_from_input(g)

        for layer in range(self.num_layers):
            # print(layer, h)
            h = self.layer_forward(layer, message)
            if self.batch_norm:
                h = self.batch_norm[layer](h)
            if layer != self.num_layers - 1:
                h = F.relu(h)
            if self.drop_ratio is not None:
                dropped_h = F.dropout(h, p=self.drop_ratio, training=self.training)
                if drop_mask is not None:
                    h = drop_mask.view(-1, 1) * dropped_h + torch.logical_not(drop_mask).view(-1, 1) * h
                else:
                    h = dropped_h
            message = self.build_message_from_output(g, h)
            h_list.append(h)

        if self.JK == "last":
            repr = h_list[-1]
        elif self.JK == "sum":
            repr = 0
            for layer in range(self.num_layers):
                repr += h_list[layer]
        elif self.JK == "mean":
            repr = 0
            for layer in range(self.num_layers):
                repr += h_list[layer]
            repr = repr/self.num_layers
        else:
            repr = h_list
        return repr


class MultiLayerMessagePassingVN(MultiLayerMessagePassing):
    def __init__(
            self,
            num_layers,
            inp_dim,
            out_dim,
            drop_ratio=None,
            JK="last",
            batch_norm=True,
    ):
        super().__init__(
            num_layers, inp_dim, out_dim, drop_ratio, JK, batch_norm
        )

        self.virtualnode_embedding = torch.nn.Embedding(1, self.out_dim)
        torch.nn.init.constant_(self.virtualnode_embedding.weight.data, 0)

        self.virtualnode_mlp_list = torch.nn.ModuleList()
        for layer in range(self.num_layers - 1):
            self.virtualnode_mlp_list.append(
                MLP([self.out_dim, 2 * self.out_dim, self.out_dim])
            )

    def forward(self, g):
        h_list = []

        message = self.build_message_from_input(g)

        vnode_embed = self.virtualnode_embedding(
            torch.zeros(g.batch_size, dtype=torch.int).to(g.device)
        )

        batch_node_segment = torch.arange(
            g.batch_size, dtype=torch.long, device=g.device
        ).repeat_interleave(g.batch_num_nodes())

        for layer in range(self.num_layers):
            # print(layer, h)
            h = self.layer_forward(layer, message)
            if self.batch_norm:
                h = self.batch_norm[layer](h)
            if layer != self.num_layers - 1:
                h = F.relu(h)
            if self.drop_ratio is not None:
                h = F.dropout(h, p=self.drop_ratio, training=self.training)
            message = self.build_message_from_output(g, h)
            h_list.append(h)

            if layer < self.num_layers - 1:
                vnode_emb_temp = (
                        scatter(
                            h, batch_node_segment, dim=0, dim_size=g.batch_size
                        )
                        + vnode_embed
                )

                vnode_embed = F.dropout(
                    self.virtualnode_mlp_list[layer](vnode_emb_temp),
                    self.drop_ratio,
                    training=self.training,
                )

        if self.JK == "last":
            repr = h_list[-1]
        elif self.JK == "sum":
            repr = 0
            for layer in range(self.num_layers):
                repr += h_list[layer]
        elif self.JK == "cat":
            repr = torch.cat([h_list], dim=-1)
        return repr

def masked_edge_index(edge_index, edge_mask):
    if isinstance(edge_index, torch.Tensor):
        return edge_index[:, edge_mask]

#RGCN
class RGCNEdgeConv(MessagePassing):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_relations: int,
        aggr: str = "mean",
        **kwargs,
    ):
        kwargs.setdefault("aggr", aggr)
        super().__init__(**kwargs)  # "Add" aggregation (Step 5).
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_relations = num_relations

        self.weight = Parameter(
            torch.empty(self.num_relations, in_channels, out_channels)
        )

        self.root = Parameter(torch.empty(in_channels, out_channels))
        self.bias = Parameter(torch.empty(out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        glorot(self.weight)
        glorot(self.root)
        zeros(self.bias)

    def forward(
        self,
        x: OptTensor,
        xe: OptTensor,
        edge_index: Adj,
        edge_type: OptTensor = None,
    ):
        out = torch.zeros(x.size(0), self.out_channels, device=x.device)
        for i in range(self.num_relations):
            edge_mask = edge_type == i
            tmp = masked_edge_index(edge_index, edge_mask)

            h = self.propagate(tmp, x=x, xe=xe[edge_mask])
            out += h @ self.weight[i]

        out += x @ self.root
        out += self.bias

        return out

    def message(self, x_j, xe):
        # x_j has shape [E, out_channels]

        # Step 4: Normalize node features.
        return (x_j + xe).relu()

#APPNP单步模拟单层
class APPNPEdgeStep(MessagePassing):
    def __init__(self, alpha: float, dropout: float = 0.0, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)
        self.alpha = alpha
        self.dropout = dropout

    def forward(
        self,
        h: torch.Tensor,
        h_0: torch.Tensor,
        xe: OptTensor,
        edge_index: Adj,
        edge_type: OptTensor = None,
    ) -> torch.Tensor:
        # Add self-loops
        edge_index_loop, _ = add_self_loops(edge_index, num_nodes=h.size(0))

        # Compute normalization
        row, col = edge_index_loop
        deg = degree(col, h.size(0), dtype=h.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Pad edge features for self-loops
        if xe is not None:
            if xe.size(-1) != h.size(-1):
                raise ValueError("Edge feature dim must match node feature dim.")
            self_loop_xe = xe.new_zeros(h.size(0), xe.size(-1))
            xe_padded = torch.cat([xe, self_loop_xe], dim=0)
        else:
            xe_padded = None

        # Apply dropout
        if self.training and self.dropout > 0:
            h = F.dropout(h, p=self.dropout, training=True)

        # Propagate
        h_prop = self.propagate(
            edge_index_loop, x=h, xe=xe_padded, norm=norm
        )

        # APPNP update: (1 - α) * propagated + α * h_0
        return (1 - self.alpha) * h_prop + self.alpha * h_0

    def message(self, x_j, xe, norm):
        msg = (x_j + xe).relu() if xe is not None else x_j.relu()
        return norm.view(-1, 1) * msg

#GIN
# class GINEEdgeConv(MessagePassing):
#     def __init__(
#         self,
#         in_channels: int,
#         out_channels: int,
#         eps: float = 0.0,
#         train_eps: bool = False,
#         aggr: str = "add",
#         nn: Optional[Callable] = None,
#         **kwargs,
#     ):
#         kwargs.setdefault("aggr", aggr)
#         super().__init__(**kwargs)
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.initial_eps = eps

#         if nn is None:
#             self.nn = Sequential(
#                 Linear(in_channels, out_channels),
#                 ReLU(),
#                 Linear(out_channels, out_channels)
#             )
#         else:
#             self.nn = nn

#         if train_eps:
#             self.eps = torch.nn.Parameter(torch.Tensor([eps]))
#         else:
#             self.register_buffer('eps', torch.Tensor([eps]))

#         self.reset_parameters()

#     def reset_parameters(self):
#         if hasattr(self.nn, 'reset_parameters'):
#             self.nn.reset_parameters()
#         else:
#             for layer in self.nn:
#                 if hasattr(layer, 'reset_parameters'):
#                     layer.reset_parameters()
#         self.eps.data.fill_(self.initial_eps)

#     def forward(
#         self,
#         x: OptTensor,
#         xe: OptTensor,
#         edge_index: Adj,
#         edge_type: OptTensor = None,  # unused, for interface consistency
#     ) -> Tensor:
#         # Perform message passing with edge features
#         out = self.propagate(edge_index, x=x, xe=xe)

#         # Add (1 + eps) * x
#         out = (1 + self.eps) * x + out

#         # Apply MLP
#         return self.nn(out)

#     def message(self, x_j: Tensor, xe: OptTensor) -> Tensor:
#         # Combine neighbor node feature and edge feature
#         if xe is not None:
#             msg = x_j + xe
#         else:
#             msg = x_j
#         return msg.relu()  # as in your RGCN template

#     def __repr__(self) -> str:
#         return f'{self.__class__.__name__}(nn={self.nn})'

#强行加入边处理后的GIN，效果有待验证
class GINEEdgeConv(MessagePassing):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        edge_dim: int = None,  # 明确指定边特征维度
        eps: float = 0.0,
        train_eps: bool = False,
        aggr: str = "add",
        nn: Optional[Callable] = None,
        **kwargs,
    ):
        kwargs.setdefault("aggr", aggr)
        super().__init__(**kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edge_dim = edge_dim
        self.initial_eps = eps

        # 为边特征创建变换层
        if edge_dim is not None:
            # 将节点和边特征映射到相同维度再相加
            self.edge_lin = Linear(edge_dim, in_channels)
        else:
            self.edge_lin = None

        #MLP定义
        if nn is None:
            self.nn = Sequential(
                Linear(in_channels, out_channels),
                ReLU(),
                Linear(out_channels, out_channels)
            )
        else:
            self.nn = nn

        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer('eps', torch.Tensor([eps]))

        self.reset_parameters()

    def forward(
        self,
        x: OptTensor,
        xe: OptTensor,
        edge_index: Adj,
        edge_type: OptTensor = None,
    ) -> Tensor:
        # 预处理边特征
        processed_xe = self.preprocess_edge_features(xe)
        
        # Perform message passing
        out = self.propagate(edge_index, x=x, xe=processed_xe)

        # Add (1 + eps) * x
        if x is not None:
            out = (1 + self.eps) * x + out

        # Apply MLP
        return self.nn(out)

    def preprocess_edge_features(self, xe: OptTensor) -> OptTensor:
        if xe is not None and self.edge_lin is not None:
            return self.edge_lin(xe)
        return xe

    def message(self, x_j: Tensor, xe: OptTensor) -> Tensor:
        # Combine neighbor node feature and processed edge feature
        if xe is not None:
            # 确保维度匹配后再相加
            if x_j.size(-1) == xe.size(-1):
                msg = x_j + xe
            else:
                # 如果维度不匹配，可以选择拼接或其他方式
                raise ValueError(f"Dimension mismatch: x_j {x_j.size()} vs xe {xe.size()}")
        else:
            msg = x_j
        # return msg.relu()
        return msg