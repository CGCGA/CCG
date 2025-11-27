import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, APPNP
from torch_geometric.utils import add_self_loops, degree
from torch_scatter import scatter
from torch_geometric.typing import Adj, OptTensor
from torch_geometric.data import Data, Batch
from utils.GNN import MultiLayerMessagePassing

class EdgeEnhancedAPPNP(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        K: int = 10,
        alpha: float = 0.1,
        dropout: float = 0.5,
        use_batch_norm: bool = False
    ):
        super().__init__()
        self.dropout = dropout
        self.use_batch_norm = use_batch_norm

        # Step 1: Edge-aware feature enhancement (e.g., using a GINE-like layer)
        self.edge_emb = nn.Linear(in_channels, hidden_channels)  # 将边特征映射到节点维度
        self.node_emb = nn.Linear(in_channels, hidden_channels)

        if use_batch_norm:
            self.bn1 = nn.BatchNorm1d(hidden_channels)

        # Step 2: MLP for initial prediction (H^{(0)})
        self.mlp = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, out_channels)
        )

        # Step 3: Standard APPNP propagation (non-parametric)
        self.propagation = APPNP(K=K, alpha=alpha, dropout=dropout)

    def forward(self, x, edge_index=None, edge_attr=None, edge_type=None):
        # 如果 x 是 Data/Batch 对象，则解包
        if isinstance(x, (Data, Batch)):
            edge_index = x.edge_index
            edge_attr = x.edge_attr if hasattr(x, 'edge_attr') else None
            edge_type = x.edge_type if hasattr(x, 'edge_type') else None
            x = x.x

        # 确保 x 不为 None
        if x is None:
            raise ValueError("Node features x must be provided.")

        # 原有逻辑不变...
        if edge_attr is None:
            edge_attr = x.new_zeros((edge_index.size(1), x.size(1)))

        x_node = self.node_emb(x)
        e_emb = self.edge_emb(edge_attr)

        edge_index_loop, e_emb_loop = add_self_loops(
            edge_index,
            edge_attr=e_emb,
            fill_value=0.0,
            num_nodes=x.size(0)
        )

        row, col = edge_index_loop
        x_j = x_node[col]
        msg = x_j + e_emb_loop
        x_enhanced = scatter(msg, row, dim=0, dim_size=x.size(0), reduce='mean')

        if self.use_batch_norm:
            x_enhanced = self.bn1(x_enhanced)

        x_enhanced = F.relu(x_enhanced)
        x_enhanced = F.dropout(x_enhanced, p=self.dropout, training=self.training)

        h0 = self.mlp(x_enhanced)
        out = self.propagation(h0, edge_index)

        return out
    
#APPNP
class StandardAPPNP(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        K: int = 10,
        alpha: float = 0.1,
        dropout: float = 0.5,
        use_batch_norm: bool = False
    ):
        super().__init__()
        self.dropout = dropout
        self.use_batch_norm = use_batch_norm

        # 标准APPNP: 直接对原始节点特征进行MLP变换
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.BatchNorm1d(hidden_channels) if use_batch_norm else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels),
            nn.BatchNorm1d(hidden_channels) if use_batch_norm else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, out_channels)
        )

        # 标准APPNP传播层
        self.propagation = APPNP(K=K, alpha=alpha, dropout=dropout)

    def forward(self, x, edge_index=None, edge_attr=None, edge_type=None):
        # 如果 x 是 Data/Batch 对象，则解包
        if isinstance(x, (Data, Batch)):
            edge_index = x.edge_index
            edge_attr = x.edge_attr if hasattr(x, 'edge_attr') else None
            edge_type = x.edge_type if hasattr(x, 'edge_type') else None
            x = x.x

        # 确保 x 不为 None
        if x is None:
            raise ValueError("Node features x must be provided.")

        # 标准APPNP: 直接对原始节点特征应用MLP
        h0 = self.mlp(x)
        
        # 使用边索引进行APPNP传播
        out = self.propagation(h0, edge_index)
        
        return out