import copy
import torch
import torch.nn.functional as F
import numpy as np
from torch.nn import Parameter, Linear, ModuleList
from torch import nn
from torch_geometric.transforms.add_positional_encoding import AddRandomWalkPE
from utils.model import SingleHeadAtt
from utils.utils import MLP

LLM_DIM = {"DB": 768, "ST": 768}

#拼接后MLP聚合嵌入
class MLPMultiGNNModel(torch.nn.Module):
    """
    集成多GNN模型，支持输出拼接融合
    """
    def __init__(self, models, llm_name, outdim, task_dim, add_rwpe=None, 
                 use_attention=True, dropout=0.0, weight_init=None, Init="PPR", **kwargs):
        super().__init__()
        assert llm_name in LLM_DIM.keys()

        self.models = ModuleList(models)  # 多个GNN模型
        self.llm_name = llm_name
        self.outdim = outdim
        self.task_dim = task_dim
        self.use_attention = use_attention
        self.num_models = len(models)
        self.K = 3

        self.Init = Init
        self.alpha = 0.2
        self.rank = 5 #矩阵尺寸
        self.sparse = True
        self.dropout = dropout

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
        
        # # 每个模型的MLP（用于处理各自的输出） 这一步应该无用
        # self.mlps = ModuleList([
        #     MLP([outdim, 2 * outdim, outdim], dropout=0.0)  # 输出维度仍为outdim
        #     for _ in range(len(models))
        # ])

        #输出的mlp
        self.mlp = MLP([outdim, 2 * outdim, outdim, task_dim], dropout=0.0)

        # 拼接后的融合MLP
        total_concat_dim = outdim * len(models)  # 所有模型输出拼接后的总维度
        self.fusion_mlp = MLP([total_concat_dim, total_concat_dim, outdim], dropout=dropout) # 此处隐藏层维度可设置//2
        # self.fusion_mlp = make_mlplayers(total_concat_dim, [total_concat_dim], False, outdim)
        self.fusion_mlp.apply(self.weights_init)
        
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


    def weights_init(self, m):
        # Xavier初始化，适合tanh/sigmoid
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

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

            # 是否有必要多一步mlp
            final_output = self.mlp(final_output)
            
            return final_output

    def _forward_single_model_for_concat(self, g, model_idx):
        """获取单个模型的输出用于拼接"""
        model = self.models[model_idx]
        model_type = self.model_types[model_idx]
        
        # 复制图数据以避免修改原始数据
        # g_copy = copy.deepcopy(g)
        g_copy = g.clone()
        
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
        
        # 无必要再经过一层mlp
        # # 通过各自的MLP处理
        # processed_emb = mlp(emb)  # [num_nodes, outdim]
        
        return emb

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

    def freeze_gnn_parameters(self):
        """冻结所有GNN相关参数"""
        for p in self.llm_proj.parameters():
            p.requires_grad = False
        
        for model in self.models:
            for p in model.parameters():
                p.requires_grad = False
        
        # # 冻结各个模型的MLP
        # for mlp in self.mlps:
        #     for p in mlp.parameters():
        #         p.requires_grad = False
        
        if self.use_attention:
            for p in self.att.parameters():
                p.requires_grad = False

        # 只有当fusion_mlp存在时才冻结它
        if hasattr(self, 'fusion_mlp') and self.fusion_mlp is not None:
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

#下游版MLP2
#继承方法 m2 = DownMLP2multiGNNModel(source_instance=m1)
class DownMLPMultiGNNModel(MLPMultiGNNModel):
    def __init__(self, models, llm_name, outdim, task_dim, add_rwpe=None, 
                 use_attention=True, dropout=0.0, weight_init=None, Init="PPR", 
                 source_instance=None, **kwargs):
        
        if source_instance is not None:
            # 从现有实例复制参数
            super().__init__(
                models=source_instance.models,
                llm_name=source_instance.llm_name,
                outdim=source_instance.outdim,
                task_dim=source_instance.task_dim,
                add_rwpe=None,
                use_attention=source_instance.use_attention,
                dropout=source_instance.dropout,
                weight_init=weight_init,
                Init=source_instance.Init,
                **kwargs
            )
            # 复制状态字典
            if hasattr(source_instance, 'state_dict'):
                self.load_state_dict(source_instance.state_dict())
        else:
            # 正常的初始化路径
            super().__init__(models, llm_name, outdim, task_dim, add_rwpe, 
                            use_attention, dropout, weight_init, Init, **kwargs)
            
        self.rank = 5
        self.K = 3
            
        if self.Init == 'PPR':
            # PPR-like
            TEMP = self.alpha * (1 - self.alpha) ** np.arange(self.K + 1)
            TEMP[-1] = (1 - self.alpha) ** self.K
            TEMP = torch.tensor([TEMP for i in range(self.rank)])
        elif self.Init == 'Random':
            # Random
            bound = np.sqrt(3 / (self.K + 1))
            TEMP = np.random.uniform(-bound, bound, self.K + 1)
            TEMP = TEMP / np.sum(np.abs(TEMP))
            TEMP = np.array([TEMP for i in range(self.rank)])
        elif self.Init== 'Fix':
            TEMP = np.ones(self.K + 1)
            TEMP = np.array([TEMP for i in range(self.rank)])
        elif self.Init == 'Mine':
            TEMP = []
            para = torch.ones([self.rank, self.K + 1])
            TEMP = torch.nn.init.xavier_normal_(para)
        elif self.Init == 'Mine_PPR':
            TEMP = self.alpha * (1 - self.alpha) ** np.arange(self.K + 1)  # 创建一个数组 其中每个元素的值根据公式 alpha*(1-alpha)**i 计算得出
            TEMP[-1] = (1 - self.alpha) ** self.K
            TEMP = torch.tensor(
                np.array([TEMP] * self.rank))  # 将 NumPy 数组 TEMP 转换为 PyTorch 张量，并使用 rank 参数指定张量的复制次数，以匹配模型中 gamma 参数的形状。

        # 微调的矩阵gamma
        self.gamma = Parameter(TEMP.float())

        proj_list = []
        for _ in range(self.K + 1):
            proj_list.append(Linear(self.outdim, self.rank))
        self.proj_list = ModuleList(proj_list)

        # self.MLP = MLP([self.outdim, self.outdim, task_dim], dropout=dropout)

        self.freeze_gnn_parameters()

        
        
        
    def forward(self, g, model_idx=None):
        g = self.initial_projection(g)

        if self.rwpe is not None:
            # 只对RWPE进行归一化，不使用no_grad
            rwpe_norm = self.rwpe_normalization(g.rwpe)
            g.x = torch.cat([g.x, rwpe_norm], dim=-1)       #节点特征拼接RWPE
            if hasattr(g, 'edge_attr') and g.edge_attr is not None:     #边特征拼接RWPE
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
            # 获取所有模型的输出
            all_outputs = []
            for i, model in enumerate(self.models):
                model_output = self._forward_single_model_for_concat(g, i)
                all_outputs.append(model_output)

            embed1 = all_outputs[0]
            embed2 = all_outputs[1]
            embed3 = all_outputs[2]
            h_0 = torch.tanh(self.proj_list[0](embed1))
            h_1 = torch.tanh(self.proj_list[1](embed2))
            h_2 = torch.tanh(self.proj_list[2](embed3))

            gamma_0 = self.gamma[:, 0].unsqueeze(dim=-1)  # gamma_0=[3,1]
            gamma_1 = self.gamma[:, 1].unsqueeze(dim=-1)  # 形状（(rank, K+1）
            gamma_2 = self.gamma[:, 2].unsqueeze(dim=-1)  # 形状（(rank, K+1）
            eta_0 = torch.matmul(h_0, gamma_0) / self.rank  # 低秩分解[2708, 1]
            eta_1 = torch.matmul(h_1, gamma_1) / self.rank
            eta_2 = torch.matmul(h_2, gamma_2) / self.rank

            hidden = torch.matmul(embed1.unsqueeze(dim=-1), eta_0.unsqueeze(dim=-1)).squeeze(dim=-1)
            hidden = hidden + torch.matmul(embed2.unsqueeze(dim=-1), eta_1.unsqueeze(dim=-1)).squeeze(dim=-1)
            hidden = hidden + torch.matmul(embed3.unsqueeze(dim=-1), eta_2.unsqueeze(dim=-1)).squeeze(dim=-1)
            h_p_1=hidden

            target_embeddings = h_p_1[g.true_nodes_mask]

            #最终输出
            final_output = self.mlp(target_embeddings)
            
            return final_output