import gc
import os
import torch
import torch.nn.functional as F
import numpy as np
from torch.nn import Parameter, Linear, ModuleList
import bitsandbytes as bnb
from accelerate.hooks import remove_hook_from_module
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch import Tensor
from torch import nn
from torch_geometric.nn.pool import global_add_pool
from torch_geometric.transforms.add_positional_encoding import AddRandomWalkPE
from torch_geometric.utils import (to_scipy_sparse_matrix, scatter, )
from torchmetrics import AveragePrecision, AUROC
from tqdm.autonotebook import trange
from transformers import BitsAndBytesConfig
from transformers import (LlamaForCausalLM, LlamaTokenizer, AutoTokenizer, AutoModel)
from utils.GNN import APPNPEdgeStep, GINEEdgeConv, MultiLayerMessagePassing, RGCNEdgeConv
from torch_geometric.nn import MessagePassing
from utils.model import SingleHeadAtt
from utils.utils import MLP

LLM_DIM = {"DB": 768, "ST": 768}

class MultiGNNModel(torch.nn.Module):
    """
    多GNN模型，使用单个LLM嵌入和多个GNN模型
    """
    def __init__(self, models, llm_name, outdim, task_dim, add_rwpe=None, use_attention=False, dropout=0.0, **kwargs):
        super().__init__()
        assert llm_name in LLM_DIM.keys()
        
        self.models = torch.nn.ModuleList(models)  # 多个GNN模型
        self.llm_name = llm_name
        self.outdim = outdim
        self.use_attention = use_attention
        
        # LLM投影层（共享）
        self.llm_proj = nn.Linear(LLM_DIM[llm_name], outdim)
        
        # 每个模型的MLP（可选，也可以共享）
        self.mlps = nn.ModuleList([
            MLP([outdim, 2 * outdim, outdim, task_dim], dropout=0.0) 
            for _ in range(len(models))
        ])
        
        # 注意力机制（如果需要）
        if use_attention:
            self.att = SingleHeadAtt(outdim)
        
        # RWPE相关
        if add_rwpe is not None:
            self.rwpe = AddRandomWalkPE(add_rwpe)
            self.edge_rwpe_prior = torch.nn.Parameter(
                torch.zeros((1, add_rwpe))
            )
            torch.nn.init.xavier_uniform_(self.edge_rwpe_prior)
            self.rwpe_normalization = torch.nn.BatchNorm1d(add_rwpe)
            self.walk_length = add_rwpe
        else:
            self.rwpe = None

    def initial_projection(self, g):
        """初始投影，将LLM嵌入映射到模型维度"""
        g.x = self.llm_proj(g.x)
        if hasattr(g, 'edge_attr') and g.edge_attr is not None:
            g.edge_attr = self.llm_proj(g.edge_attr)
        return g

    def forward(self, g, model_idx=None):
        """
        前向传播
        
        Args:
            g: 图数据
            model_idx: 指定使用的模型索引，如果为None则返回所有模型的输出列表
        """
        g = self.initial_projection(g)

        if self.rwpe is not None:
            with torch.no_grad():
                rwpe_norm = self.rwpe_normalization(g.rwpe)
                g.x = torch.cat([g.x, rwpe_norm], dim=-1)
                if hasattr(g, 'edge_attr') and g.edge_attr is not None:
                    g.edge_attr = torch.cat(
                        [
                            g.edge_attr,
                            self.edge_rwpe_prior.repeat(len(g.edge_attr), 1),
                        ],
                        dim=-1,
                    )
        
        if model_idx is not None:
            # 只使用指定的模型
            return self._forward_single_model(g, model_idx)
        else:
            # 返回所有模型的输出
            outputs = []
            for i, model in enumerate(self.models):
                output = self._forward_single_model(g, i)
                outputs.append(output)
            return outputs

    def _forward_single_model(self, g, model_idx):
        """单个模型的前向传播"""
        model = self.models[model_idx]
        mlp = self.mlps[model_idx]
        
        if self.use_attention:
            # 使用注意力机制
            emb = torch.stack(model(g), dim=1)  # 假设model返回多层嵌入
            query = g.x.unsqueeze(1)
            emb = self.att(emb, query, emb)[0].squeeze()
        else:
            # 直接使用模型输出
            emb = model(g)
        
        # 提取真实节点的嵌入并进行分类
        class_emb = emb[g.true_nodes_mask]
        res = mlp(class_emb)
        return res

    def get_all_outputs(self, g):
        """获取所有模型的输出"""
        g = self.initial_projection(g)

        if self.rwpe is not None:
            with torch.no_grad():
                rwpe_norm = self.rwpe_normalization(g.rwpe)
                g.x = torch.cat([g.x, rwpe_norm], dim=-1)
                if hasattr(g, 'edge_attr') and g.edge_attr is not None:
                    g.edge_attr = torch.cat(
                        [
                            g.edge_attr,
                            self.edge_rwpe_prior.repeat(len(g.edge_attr), 1),
                        ],
                        dim=-1,
                    )
        
        outputs = []
        for i, model in enumerate(self.models):
            if self.use_attention:
                emb = torch.stack(model(g), dim=1)
                query = g.x.unsqueeze(1)
                emb = self.att(emb, query, emb)[0].squeeze()
            else:
                emb = model(g)
            
            class_emb = emb[g.true_nodes_mask]
            res = self.mlps[i](class_emb)
            outputs.append(res)
        
        return outputs

    def freeze_gnn_parameters(self):
        """冻结所有GNN相关参数"""
        for p in self.llm_proj.parameters():
            p.requires_grad = False
        
        for model in self.models:
            for p in model.parameters():
                p.requires_grad = False
        
        for mlp in self.mlps:
            for p in mlp.parameters():
                p.requires_grad = False
        
        if self.use_attention:
            for p in self.att.parameters():
                p.requires_grad = False


class EnsembleMultiGNNModel(torch.nn.Module):
    """
    集成多GNN模型，支持权重聚合
    """
    def __init__(self, models, llm_name, outdim, task_dim, add_rwpe=None, 
                 use_attention=True, dropout=0.0, weight_init=None, **kwargs):
        super().__init__()
        assert llm_name in LLM_DIM.keys()
        
        self.models = torch.nn.ModuleList(models)  # 多个GNN模型
        self.llm_name = llm_name
        self.outdim = outdim
        self.use_attention = use_attention

        # 为每个模型标记类型（可选）
        self.model_types = []  # 'layered' 或 'iterative'
        for model in models:
            # 可以通过检查模型类型或配置来确定
            if hasattr(model, 'num_layers'):  # 逐层GNN
                self.model_types.append('layered')
            else:  # 假设是迭代GNN
                self.model_types.append('iterative')
        
        # LLM投影层（共享）
        self.llm_proj = nn.Linear(LLM_DIM[llm_name], outdim)
        
        # 每个模型的MLP
        self.mlps = nn.ModuleList([
            MLP([outdim, 2 * outdim, outdim, task_dim], dropout=0.0) 
            for _ in range(len(models))
        ])
        
        # 注意力机制
        if use_attention:
            self.att = SingleHeadAtt(outdim)
        
        # RWPE相关
        if add_rwpe is not None:
            self.rwpe = AddRandomWalkPE(add_rwpe)
            self.edge_rwpe_prior = torch.nn.Parameter(
                torch.zeros((1, add_rwpe))
            )
            torch.nn.init.xavier_uniform_(self.edge_rwpe_prior)
            self.rwpe_normalization = torch.nn.BatchNorm1d(add_rwpe)
            self.walk_length = add_rwpe
        else:
            self.rwpe = None
        
        # 集成权重
        if weight_init is None:
            weight_init = [1.0 / len(models)] * len(models)
        # 使用更大的初始值，避免梯度消失
        self.raw_weights = nn.Parameter(torch.tensor(weight_init, dtype=torch.float32) * 10.0)
        self.normalize_weights = True

    def initial_projection(self, g):
        g.x = self.llm_proj(g.x)
        if hasattr(g, 'edge_attr') and g.edge_attr is not None:
            g.edge_attr = self.llm_proj(g.edge_attr)
        return g

    def forward(self, g, model_idx=None):
        g = self.initial_projection(g)

        if self.rwpe is not None:
            # 只对RWPE进行归一化，不使用no_grad
            rwpe_norm = self.rwpe_normalization(g.rwpe)
            g.x = torch.cat([g.x, rwpe_norm], dim=-1)
            if hasattr(g, 'edge_attr') and g.edge_attr is not None:
                g.edge_attr = torch.cat(
                    [
                        g.edge_attr,
                        self.edge_rwpe_prior.repeat(len(g.edge_attr), 1),
                    ],
                    dim=-1,
                )
        
        if model_idx is not None:
            return self._forward_single_model(g, model_idx)
        else:
            outputs = []
            for i, model in enumerate(self.models):
                output = self._forward_single_model(g, i)
                outputs.append(output)
            
            # 使用可学习的权重进行加权聚合
            # 获取归一化的权重（确保权重为正且和为1）
            if self.normalize_weights:
                # 使用softmax，保持梯度流
                normalized_weights = torch.softmax(self.raw_weights, dim=0)
            else:
                # 使用sigmoid，确保正值
                normalized_weights = torch.sigmoid(self.raw_weights)
            
            # 计算加权输出
            final_output = torch.zeros_like(outputs[0])
            for i, output in enumerate(outputs):
                final_output += output * normalized_weights[i]
            
            return final_output

    def _forward_single_model(self, g, model_idx):
        model = self.models[model_idx]
        mlp = self.mlps[model_idx]
        model_type = self.model_types[model_idx]
        
        if self.use_attention and model_type == 'layered':
            # 只对逐层GNN使用注意力
            emb_list = model(g)
            if isinstance(emb_list, list) and len(emb_list) > 1:
                # 多层输出，使用注意力
                emb = torch.stack(emb_list, dim=1)
                query = g.x.unsqueeze(1)
                emb = self.att(emb, query, emb)[0].squeeze()
            else:
                # 单层输出或非列表输出
                emb = emb_list if not isinstance(emb_list, list) else emb_list[-1]
        else:
            # 不使用注意力或迭代GNN
            emb = model(g)
            if isinstance(emb, list):
                emb = emb[-1]  # 取最后一层（如果是逐层GNN）
        
        class_emb = emb[g.true_nodes_mask]
        res = mlp(class_emb)
        return res

    def get_individual_outputs(self, g):
        """获取每个模型的独立输出"""
        g = self.initial_projection(g)

        if self.rwpe is not None:
            rwpe_norm = self.rwpe_normalization(g.rwpe)
            g.x = torch.cat([g.x, rwpe_norm], dim=-1)
            if hasattr(g, 'edge_attr') and g.edge_attr is not None:
                g.edge_attr = torch.cat(
                    [
                        g.edge_attr,
                        self.edge_rwpe_prior.repeat(len(g.edge_attr), 1),
                    ],
                    dim=-1,
                )
        
        outputs = []
        for i, model in enumerate(self.models):
            if self.use_attention:
                emb = torch.stack(model(g), dim=1)
                query = g.x.unsqueeze(1)
                emb = self.att(emb, query, emb)[0].squeeze()
            else:
                emb = model(g)
            
            class_emb = emb[g.true_nodes_mask]
            res = self.mlps[i](class_emb)
            outputs.append(res)
        
        return outputs
    
    def get_current_weights(self):
        """获取当前的归一化权重值"""
        with torch.no_grad():
            if self.normalize_weights:
                return torch.softmax(self.raw_weights, dim=0).cpu().numpy()
            else:
                return torch.sigmoid(self.raw_weights).cpu().numpy()

    def freeze_gnn_parameters(self, freeze_weights=False):
        """冻结所有GNN相关参数"""
        for p in self.llm_proj.parameters():
            p.requires_grad = False
        
        for model in self.models:
            for p in model.parameters():
                p.requires_grad = False
        
        #不冻结mlp
        # for mlp in self.mlps:
        #     for p in mlp.parameters():
        #         p.requires_grad = False
        
        if self.use_attention:
            for p in self.att.parameters():
                p.requires_grad = False

        # 根据参数决定是否冻结集成权重
        if freeze_weights:
            self.raw_weights.requires_grad = False
        else:
            self.raw_weights.requires_grad = True  # 确保权重可以被训练

        print("freeze over")

    def unfreeze_weights(self):
        """解冻集成权重"""
        self.raw_weights.requires_grad = True

    def freeze_all_except_weights(self):
        """冻结所有参数除了集成权重"""
        self.freeze_gnn_parameters(freeze_weights=False)  # 不冻结weights
        # 其他参数保持冻结状态，weights保持可训练



#softmax
class SoftmaxMultiGNNModel(torch.nn.Module):
    """
    集成多GNN模型，支持softmax注意力聚合
    """
    def __init__(self, models, llm_name, outdim, task_dim, add_rwpe=None, 
                 use_attention=True, dropout=0.0, weight_init=None, **kwargs):
        super().__init__()
        assert llm_name in LLM_DIM.keys()
        
        self.models = torch.nn.ModuleList(models)  # 多个GNN模型
        self.llm_name = llm_name
        self.outdim = outdim
        self.task_dim = task_dim  # 添加task_dim属性
        self.use_attention = use_attention
        self.num_models = len(models)

        # 为每个模型标记类型（可选）
        self.model_types = []  # 'layered' 或 'iterative'
        for model in models:
            # 可以通过检查模型类型或配置来确定
            if hasattr(model, 'num_layers'):  # 逐层GNN
                self.model_types.append('layered')
            else:  # 假设是迭代GNN
                self.model_types.append('iterative')
        
        # LLM投影层（共享）
        self.llm_proj = nn.Linear(LLM_DIM[llm_name], outdim)
        
        # 每个模型的MLP - 确保输出维度为task_dim
        self.mlps = nn.ModuleList([
            MLP([outdim, 2 * outdim, outdim, task_dim], dropout=0.0) 
            for _ in range(len(models))
        ])
        
        # 注意力机制 - 用于逐层GNN内部
        if use_attention:
            self.att = SingleHeadAtt(outdim)
        
        # RWPE相关
        if add_rwpe is not None:
            self.rwpe = AddRandomWalkPE(add_rwpe)
            self.edge_rwpe_prior = torch.nn.Parameter(
                torch.zeros((1, add_rwpe))
            )
            torch.nn.init.xavier_uniform_(self.edge_rwpe_prior)
            self.rwpe_normalization = torch.nn.BatchNorm1d(add_rwpe)
            self.walk_length = add_rwpe
        else:
            self.rwpe = None
        
        # Softmax聚合权重 - 使用可学习的权重向量
        if weight_init is None:
            weight_init = [1.0 / len(models)] * len(models)
        # 初始化原始权重参数（未归一化）
        self.model_weights = nn.Parameter(torch.tensor(weight_init, dtype=torch.float32))
        
        # 可选：添加上下文感知的动态权重计算
        self.use_dynamic_weights = kwargs.get('use_dynamic_weights', False)
        if self.use_dynamic_weights:
            # 为动态权重计算添加一个小的网络
            self.weight_predictor = nn.Sequential(
                nn.Linear(outdim, outdim // 2),
                nn.ReLU(),
                nn.Linear(outdim // 2, len(models))
            )

    def initial_projection(self, g):
        g.x = self.llm_proj(g.x)
        if hasattr(g, 'edge_attr') and g.edge_attr is not None:
            g.edge_attr = self.llm_proj(g.edge_attr)
        return g

    def compute_model_weights(self, g=None):
        """计算模型权重 - 静态或动态"""
        if self.use_dynamic_weights and g is not None:
            # 基于图特征动态计算权重
            # 使用图的全局特征或节点特征的聚合
            global_feat = g.x.mean(dim=0, keepdim=True)  # 或者使用其他聚合方式
            dynamic_weights = self.weight_predictor(global_feat).squeeze(0)
            return torch.softmax(dynamic_weights, dim=0)
        else:
            # 静态可学习权重
            return torch.softmax(self.model_weights, dim=0)

    def forward(self, g, model_idx=None):
        g = self.initial_projection(g)

        if self.rwpe is not None:
            # 只对RWPE进行归一化，不使用no_grad
            rwpe_norm = self.rwpe_normalization(g.rwpe)
            g.x = torch.cat([g.x, rwpe_norm], dim=-1)
            if hasattr(g, 'edge_attr') and g.edge_attr is not None:
                g.edge_attr = torch.cat(
                    [
                        g.edge_attr,
                        self.edge_rwpe_prior.repeat(len(g.edge_attr), 1),
                    ],
                    dim=-1,
                )
        
        if model_idx is not None:
            return self._forward_single_model(g, model_idx)
        else:
            # 获取所有模型的输出 - 每个输出是 [num_true_nodes, task_dim]
            outputs = []
            for i, model in enumerate(self.models):
                output = self._forward_single_model(g, i)
                outputs.append(output)  # [num_true_nodes, task_dim]
            
            # 将输出堆叠成 [num_true_nodes, num_models, task_dim]
            outputs = torch.stack(outputs, dim=1)  # [num_true_nodes, num_models, task_dim]
            
            # 计算softmax权重
            weights = self.compute_model_weights(g)  # [num_models]
            
            # 使用softmax权重进行加权聚合
            # weights: [num_models] -> [num_true_nodes, num_models, 1] for broadcasting
            expanded_weights = weights.unsqueeze(0).unsqueeze(-1).expand(
                outputs.size(0), -1, -1
            )  # [num_true_nodes, num_models, 1]
            
            final_output = torch.sum(outputs * expanded_weights, dim=1)  # [num_true_nodes, task_dim]
            
            return final_output

    def _forward_single_model(self, g, model_idx):
        model = self.models[model_idx]
        mlp = self.mlps[model_idx]
        model_type = self.model_types[model_idx]
        
        if self.use_attention and model_type == 'layered':
            # 只对逐层GNN使用注意力
            emb_list = model(g)
            if isinstance(emb_list, list) and len(emb_list) > 1:
                # 多层输出，使用注意力
                emb = torch.stack(emb_list, dim=1)
                query = g.x.unsqueeze(1)
                emb = self.att(emb, query, emb)[0].squeeze()
            else:
                # 单层输出或非列表输出
                emb = emb_list if not isinstance(emb_list, list) else emb_list[-1]
        else:
            # 不使用注意力或迭代GNN
            emb = model(g)
            if isinstance(emb, list):
                emb = emb[-1]  # 取最后一层（如果是逐层GNN）
        
        class_emb = emb[g.true_nodes_mask]  # [num_true_nodes, hidden_dim]
        res = mlp(class_emb)  # [num_true_nodes, task_dim]
        return res

    def get_individual_outputs(self, g):
        """获取每个模型的独立输出"""
        g = self.initial_projection(g)

        if self.rwpe is not None:
            rwpe_norm = self.rwpe_normalization(g.rwpe)
            g.x = torch.cat([g.x, rwpe_norm], dim=-1)
            if hasattr(g, 'edge_attr') and g.edge_attr is not None:
                g.edge_attr = torch.cat(
                    [
                        g.edge_attr,
                        self.edge_rwpe_prior.repeat(len(g.edge_attr), 1),
                    ],
                    dim=-1,
                )
        
        outputs = []
        for i, model in enumerate(self.models):
            output = self._forward_single_model(g, i)  # [num_true_nodes, task_dim]
            outputs.append(output)
        
        return outputs
    
    def get_current_weights(self):
        """获取当前的softmax归一化权重值"""
        with torch.no_grad():
            weights = torch.softmax(self.model_weights, dim=0)
            return weights.cpu().numpy()

    def get_raw_weights(self):
        """获取原始权重值（未经过softmax）"""
        with torch.no_grad():
            return self.model_weights.cpu().numpy()

    def freeze_gnn_parameters(self, freeze_weights=False):
        """冻结所有GNN相关参数"""
        for p in self.llm_proj.parameters():
            p.requires_grad = False
        
        for model in self.models:
            for p in model.parameters():
                p.requires_grad = False
        
        # 不冻结mlp
        # for mlp in self.mlps:
        #     for p in mlp.parameters():
        #         p.requires_grad = False
        
        if self.use_attention:
            for p in self.att.parameters():
                p.requires_grad = False

        # 根据参数决定是否冻结集成权重
        if freeze_weights:
            self.model_weights.requires_grad = False
        else:
            self.model_weights.requires_grad = True  # 确保权重可以被训练

        # 如果使用动态权重，也要处理weight_predictor
        if self.use_dynamic_weights:
            for p in self.weight_predictor.parameters():
                p.requires_grad = not freeze_weights

        print("freeze over")

    def unfreeze_weights(self):
        """解冻集成权重"""
        self.model_weights.requires_grad = True
        if self.use_dynamic_weights:
            for p in self.weight_predictor.parameters():
                p.requires_grad = True

    def freeze_all_except_weights(self):
        """冻结所有参数除了集成权重"""
        self.freeze_gnn_parameters(freeze_weights=False)  # 不冻结weights
        # 其他参数保持冻结状态，weights保持可训练

    def get_num_classes(self):
        """返回类别数量，用于验证"""
        return self.task_dim

#维度拼接再MLP映射
class MLPmultiGNNModel(torch.nn.Module):
    """
    集成多GNN模型，支持输出拼接融合
    """
    def __init__(self, models, llm_name, outdim, task_dim, add_rwpe=None, 
                 use_attention=True, dropout=0.0, weight_init=None, **kwargs):
        super().__init__()
        assert llm_name in LLM_DIM.keys()
        
        self.models = torch.nn.ModuleList(models)  # 多个GNN模型
        self.llm_name = llm_name
        self.outdim = outdim
        self.use_attention = use_attention
        self.num_models = len(models)

        # 为每个模型标记类型（可选）
        self.model_types = []  # 'layered' 或 'iterative'
        for model in models:
            # 可以通过检查模型类型或配置来确定
            if hasattr(model, 'num_layers'):  # 逐层GNN
                self.model_types.append('layered')
            else:  # 假设是迭代GNN
                self.model_types.append('iterative')
        
        # LLM投影层（共享）
        self.llm_proj = nn.Linear(LLM_DIM[llm_name], outdim)
        
        # 每个模型的MLP（用于处理各自的输出）
        self.mlps = nn.ModuleList([
            MLP([outdim, 2 * outdim, outdim], dropout=0.0)  # 输出维度仍为outdim
            for _ in range(len(models))
        ])
        
        # 拼接后的融合MLP
        total_concat_dim = outdim * len(models)  # 所有模型输出拼接后的总维度
        self.fusion_mlp = MLP([total_concat_dim, total_concat_dim // 2, task_dim], dropout=dropout)
        
        # 注意力机制（可选，用于逐层GNN）
        if use_attention:
            self.att = SingleHeadAtt(outdim)
        
        # RWPE相关
        if add_rwpe is not None:
            self.rwpe = AddRandomWalkPE(add_rwpe)
            self.edge_rwpe_prior = torch.nn.Parameter(
                torch.zeros((1, add_rwpe))
            )
            torch.nn.init.xavier_uniform_(self.edge_rwpe_prior)
            self.rwpe_normalization = torch.nn.BatchNorm1d(add_rwpe)
            self.walk_length = add_rwpe
        else:
            self.rwpe = None

    def initial_projection(self, g):
        g.x = self.llm_proj(g.x)
        if hasattr(g, 'edge_attr') and g.edge_attr is not None:
            g.edge_attr = self.llm_proj(g.edge_attr)
        return g

    def forward(self, g, model_idx=None):
        g = self.initial_projection(g)

        if self.rwpe is not None:
            # 只对RWPE进行归一化，不使用no_grad
            rwpe_norm = self.rwpe_normalization(g.rwpe)
            g.x = torch.cat([g.x, rwpe_norm], dim=-1)
            if hasattr(g, 'edge_attr') and g.edge_attr is not None:
                g.edge_attr = torch.cat(
                    [
                        g.edge_attr,
                        self.edge_rwpe_prior.repeat(len(g.edge_attr), 1),
                    ],
                    dim=-1,
                )
        
        if model_idx is not None:
            # 如果指定了单个模型，返回该模型的处理结果
            return self._forward_single_model(g, model_idx)
        else:
            # 获取所有模型的输出并拼接
            all_outputs = []
            for i, model in enumerate(self.models):
                model_output = self._forward_single_model_for_concat(g, i)
                all_outputs.append(model_output)
            
            # 拼接所有模型的输出
            concatenated_output = torch.cat(all_outputs, dim=-1)  # [num_nodes, total_concat_dim]
            
            # 获取目标节点的拼接特征
            target_embeddings = concatenated_output[g.true_nodes_mask]  # [num_target_nodes, total_concat_dim]
            
            # 通过融合MLP得到最终输出
            final_output = self.fusion_mlp(target_embeddings)
            
            return final_output

    def _forward_single_model_for_concat(self, g, model_idx):
        """获取单个模型的输出用于拼接"""
        model = self.models[model_idx]
        mlp = self.mlps[model_idx]
        model_type = self.model_types[model_idx]
        
        # 复制图数据以避免修改原始数据
        g_copy = g.clone()  # 假设图对象有clone方法，如果没有需要相应调整
        
        if self.use_attention and model_type == 'layered':
            # 只对逐层GNN使用注意力
            emb_list = model(g_copy)
            if isinstance(emb_list, list) and len(emb_list) > 1:
                # 多层输出，使用注意力
                emb = torch.stack(emb_list, dim=1)
                query = g_copy.x.unsqueeze(1)
                emb = self.att(emb, query, emb)[0].squeeze()
            else:
                # 单层输出或非列表输出
                emb = emb_list if not isinstance(emb_list, list) else emb_list[-1]
        else:
            # 不使用注意力或迭代GNN
            emb = model(g_copy)
            if isinstance(emb, list):
                emb = emb[-1]  # 取最后一层（如果是逐层GNN）
        
        # 通过各自的MLP处理
        processed_emb = mlp(emb)  # [num_nodes, outdim]
        
        return processed_emb

    def _forward_single_model(self, g, model_idx):
        """获取单个模型的最终输出（用于model_idx不为None的情况）"""
        model = self.models[model_idx]
        mlp = self.mlps[model_idx]
        model_type = self.model_types[model_idx]
        
        if self.use_attention and model_type == 'layered':
            emb_list = model(g)
            if isinstance(emb_list, list) and len(emb_list) > 1:
                emb = torch.stack(emb_list, dim=1)
                query = g.x.unsqueeze(1)
                emb = self.att(emb, query, emb)[0].squeeze()
            else:
                emb = emb_list if not isinstance(emb_list, list) else emb_list[-1]
        else:
            emb = model(g)
            if isinstance(emb, list):
                emb = emb[-1]
        
        class_emb = mlp(emb)[g.true_nodes_mask]
        return class_emb

    def get_individual_outputs(self, g):
        """获取每个模型的独立输出"""
        g = self.initial_projection(g)

        if self.rwpe is not None:
            rwpe_norm = self.rwpe_normalization(g.rwpe)
            g.x = torch.cat([g.x, rwpe_norm], dim=-1)
            if hasattr(g, 'edge_attr') and g.edge_attr is not None:
                g.edge_attr = torch.cat(
                    [
                        g.edge_attr,
                        self.edge_rwpe_prior.repeat(len(g.edge_attr), 1),
                    ],
                    dim=-1,
                )
        
        outputs = []
        for i, model in enumerate(self.models):
            model_type = self.model_types[i]
            
            if self.use_attention and model_type == 'layered':
                emb = torch.stack(model(g), dim=1)
                query = g.x.unsqueeze(1)
                emb = self.att(emb, query, emb)[0].squeeze()
            else:
                emb = model(g)
                if isinstance(emb, list):
                    emb = emb[-1]
            
            class_emb = self.mlps[i](emb)[g.true_nodes_mask]
            outputs.append(class_emb)
        
        return outputs
    
    def get_concatenated_output(self, g):
        """获取拼接后的输出（不经过最终融合MLP）"""
        g = self.initial_projection(g)

        if self.rwpe is not None:
            rwpe_norm = self.rwpe_normalization(g.rwpe)
            g.x = torch.cat([g.x, rwpe_norm], dim=-1)
            if hasattr(g, 'edge_attr') and g.edge_attr is not None:
                g.edge_attr = torch.cat(
                    [
                        g.edge_attr,
                        self.edge_rwpe_prior.repeat(len(g.edge_attr), 1),
                    ],
                    dim=-1,
                )
        
        all_outputs = []
        for i, model in enumerate(self.models):
            model_output = self._forward_single_model_for_concat(g, i)
            all_outputs.append(model_output)
        
        concatenated_output = torch.cat(all_outputs, dim=-1)
        target_embeddings = concatenated_output[g.true_nodes_mask]
        
        return target_embeddings

    def get_current_weights(self):
        """获取当前的集成权重值（这里没有可学习权重，返回None或默认值）"""
        # 由于现在是拼接融合，没有显式的权重参数
        return [1.0 / self.num_models] * self.num_models

    def freeze_gnn_parameters(self, freeze_weights=False):
        """冻结所有GNN相关参数"""
        for p in self.llm_proj.parameters():
            p.requires_grad = False
        
        for model in self.models:
            for p in model.parameters():
                p.requires_grad = False
        
        # 冻结各个模型的MLP
        for mlp in self.mlps:
            for p in mlp.parameters():
                p.requires_grad = False
        
        if self.use_attention:
            for p in self.att.parameters():
                p.requires_grad = False

        # 冻结融合MLP
        for p in self.fusion_mlp.parameters():
            p.requires_grad = False

        print("freeze over")

    def unfreeze_weights(self):
        """解冻融合MLP参数"""
        for p in self.fusion_mlp.parameters():
            p.requires_grad = True

    def freeze_all_except_weights(self):
        """冻结所有参数除了融合MLP"""
        self.freeze_gnn_parameters()
        # 解冻融合MLP参数
        for p in self.fusion_mlp.parameters():
            p.requires_grad = True
