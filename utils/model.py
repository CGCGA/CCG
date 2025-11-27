import gc
import os
import torch
import torch.nn.functional as F
import numpy as np
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
from utils.utils import MLP

LLM_DIM = {"DB": 768, "ST": 768}

#
def mean_pooling(token_embeddings, attention_mask):
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-10)

#可用GPU检测
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

#LLM class
class LLMModel(torch.nn.Module):
    """
    Large language model from transformers.
    If peft is ture, use lora with pre-defined parameter setting for efficient fine-tuning.
    quantization is set to 4bit and should be used in the most of the case to avoid OOM.
    """
    def __init__(self, llm_name, quantization=True, peft=True, cache_dir="model", max_length=500):
        super().__init__()
        assert llm_name in LLM_DIM.keys()
        self.llm_name = llm_name
        self.quantization = quantization

        self.indim = LLM_DIM[self.llm_name]
        self.cache_dir = cache_dir
        self.max_length = max_length
        model, self.tokenizer = self.get_llm_model()
        if peft:
            self.model = self.get_lora_perf(model)
        else:
            self.model = model
        self.tokenizer.padding_side = "right"
        self.tokenizer.truncation_side = 'right'

    def find_all_linear_names(self, model):
        """
        LoRA微调LLM
        """
        cls = bnb.nn.Linear4bit if self.quantization else torch.nn.Linear
        lora_module_names = set()
        for name, module in model.named_modules():
            if isinstance(module, cls):
                names = name.split('.')
                lora_module_names.add(names[0] if len(names) == 1 else names[-1])

        if 'lm_head' in lora_module_names:  # needed for 16-bit
            lora_module_names.remove('lm_head')
        return list(lora_module_names)

    def create_bnb_config(self):
        """
        quantization configuration.
        """
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        return bnb_config

    def get_lora_perf(self, model):
        """
        LoRA configuration.
        """
        target_modules = self.find_all_linear_names(model)
        config = LoraConfig(
            target_modules=target_modules,
            r=16,  # dimension of the updated matrices
            lora_alpha=16,  # parameter for scaling
            lora_dropout=0.2,  # dropout probability for layers
            bias="none",
            task_type="FEATURE_EXTRACTION",
        )
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
        model = get_peft_model(model, config)

        return model

#设置LLM
    def get_llm_model(self):
        if self.llm_name == "llama2_7b":
            model_name = "meta-llama/Llama-2-7b-hf"
            ModelClass = LlamaForCausalLM
            TokenizerClass = LlamaTokenizer

        elif self.llm_name == "llama2_13b":
            model_name = "meta-llama/Llama-2-13b-hf"
            ModelClass = LlamaForCausalLM
            TokenizerClass = LlamaTokenizer

        elif self.llm_name == "e5":
            model_name = "intfloat/e5-large-v2"
            ModelClass = AutoModel
            TokenizerClass = AutoTokenizer

        elif self.llm_name == "BERT":
            model_name = "bert-base-uncased"
            ModelClass = AutoModel
            TokenizerClass = AutoTokenizer

        elif self.llm_name == "ST":
            model_name = "sentence-transformers/multi-qa-distilbert-cos-v1"
            ModelClass = AutoModel
            TokenizerClass = AutoTokenizer

        else:
            raise ValueError(f"Unknown language model: {self.llm_name}.")
        #bnb量化， 直接训练与量化训练的区别
        if self.quantization:
            bnb_config = self.create_bnb_config()
            model = ModelClass.from_pretrained(model_name,
                                               quantization_config=bnb_config,
                                               #attn_implementation="flash_attention_2",
                                               #torch_type=torch.bfloat16,
                                               cache_dir=self.cache_dir)
        else:
            model = ModelClass.from_pretrained(model_name, cache_dir=self.cache_dir)
        #移除hook
        model = remove_hook_from_module(model, recurse=True)
        model.config.use_cache = False
        #加载模型
        tokenizer = TokenizerClass.from_pretrained(model_name, cache_dir=self.cache_dir, add_eos_token=True)
        if self.llm_name[:6] == "llama2":
            tokenizer.pad_token = tokenizer.bos_token
        return model, tokenizer

    def pooling(self, outputs, text_tokens=None):
        # if self.llm_name in ["BERT", "ST", "e5"]:
        return F.normalize(mean_pooling(outputs, text_tokens["attention_mask"]), p=2, dim=1)

        # else:
        #     return outputs[text_tokens["input_ids"] == 2] # llama2 EOS token

    def forward(self, text_tokens):
        outputs = self.model(input_ids=text_tokens["input_ids"],
                             attention_mask=text_tokens["attention_mask"],
                             output_hidden_states=True,
                             return_dict=True)["hidden_states"][-1]

        return self.pooling(outputs, text_tokens)

    def encode(self, text_tokens, pooling=False):

        with torch.no_grad():
            outputs = self.model(input_ids=text_tokens["input_ids"],
                                 attention_mask=text_tokens["attention_mask"],
                                 output_hidden_states=True,
                                 return_dict=True)["hidden_states"][-1]
            outputs = outputs.to(torch.float32)
            if pooling:
                outputs = self.pooling(outputs, text_tokens)

            return outputs, text_tokens["attention_mask"]

#LM class, 构建任务
class SentenceEncoder:
    def __init__(self, llm_name, cache_dir="../model", batch_size=1, multi_gpu=False):
        self.llm_name = llm_name
        self.device, _ = get_available_devices()
        self.batch_size = batch_size
        self.multi_gpu = multi_gpu
        self.model = LLMModel(llm_name, quantization=False, peft=False, cache_dir=cache_dir)
        self.model.to(self.device)

    def encode(self, texts, to_tensor=True):
        all_embeddings = []
        with torch.no_grad():
            for start_index in trange(0, len(texts), self.batch_size, desc="Batches", disable=False, ):
                sentences_batch = texts[start_index: start_index + self.batch_size]
                text_tokens = self.model.tokenizer(sentences_batch, return_tensors="pt", padding="longest", truncation=True,
                                           max_length=500).to(self.device)
                embeddings, _ = self.model.encode(text_tokens, pooling=True)
                embeddings = embeddings.cpu()
                all_embeddings.append(embeddings)
        all_embeddings = torch.cat(all_embeddings, dim=0)
        if not to_tensor:
            all_embeddings = all_embeddings.numpy()

        return all_embeddings

    def flush_model(self):
        # delete llm from gpu to save GPU memory
        if self.model is not None:
            self.model = None
        gc.collect()
        torch.cuda.empty_cache()

#RGCN
class PyGRGCNEdge(MultiLayerMessagePassing):
    def __init__(
        self,
        num_layers: int,
        num_rels: int,
        inp_dim: int,
        out_dim: int,
        drop_ratio=0,
        JK="last",
        batch_norm=True,
    ):
        super().__init__(
            num_layers, inp_dim, out_dim, drop_ratio, JK, batch_norm
        )
        self.num_rels = num_rels
        self.build_layers()

    def build_input_layer(self):
        return RGCNEdgeConv(self.inp_dim, self.out_dim, self.num_rels)

    def build_hidden_layer(self):
        return RGCNEdgeConv(self.inp_dim, self.out_dim, self.num_rels)

    def build_message_from_input(self, g):
        return {
            "g": g.edge_index,
            "h": g.x,
            "e": g.edge_type,
            "he": g.edge_attr,
        }

    def build_message_from_output(self, g, h):
        return {"g": g.edge_index, "h": h, "e": g.edge_type, "he": g.edge_attr}

    def layer_forward(self, layer, message):
        return self.conv[layer](
            message["h"], message["he"], message["g"], message["e"]
        )

#APPNP   
class PyGAPPNPEdge(MultiLayerMessagePassing):
    def __init__(
        self,
        num_layers: int,      # = K propagation steps
        num_rels: int,        # kept for interface compatibility (unused)
        inp_dim: int,
        out_dim: int,
        drop_ratio=0,
        JK="last",
        batch_norm=False,     # APPNP typically doesn't use BN in propagation
        alpha=0.1,
    ):
        super().__init__(
            num_layers, inp_dim, out_dim, drop_ratio, JK, batch_norm
        )
        self.alpha = alpha
        self.build_layers()

    def build_input_layer(self):
        return APPNPEdgeStep(alpha=self.alpha, dropout=self.drop_ratio)

    def build_hidden_layer(self):
        return APPNPEdgeStep(alpha=self.alpha, dropout=self.drop_ratio)

    def build_message_from_input(self, g):
        # Store initial representation h_0 = g.x
        return {
            "g": g.edge_index,
            "h": g.x,
            "h_0": g.x,          # <-- critical: pass initial features
            "e": getattr(g, "edge_type", None),
            "he": getattr(g, "edge_attr", None),
        }

    def build_message_from_output(self, g, h):
        # Keep h_0 unchanged across layers
        return {
            "g": g.edge_index,
            "h": h,
            "h_0": g.x,          # <-- always use original input as h_0
            "e": getattr(g, "edge_type", None),
            "he": getattr(g, "edge_attr", None),
        }

    def layer_forward(self, layer, message):
        return self.conv[layer](
            h=message["h"],       # current representation
            h_0=message["h_0"],   # initial representation (fixed)
            xe=message["he"],
            edge_index=message["g"],
            edge_type=message["e"]
        )

#GIN
class PyGGINEEdge(MultiLayerMessagePassing):
    def __init__(
        self,
        num_layers: int,
        num_rels: int,        # kept for interface compatibility (unused in GIN)
        inp_dim: int,
        out_dim: int,
        drop_ratio=0,
        JK="last",
        batch_norm=True,
        eps: float = 0.0,
        train_eps: bool = False,
    ):
        super().__init__(
            num_layers, inp_dim, out_dim, drop_ratio, JK, batch_norm
        )
        self.num_rels = num_rels  # unused, but kept for API consistency
        self.eps = eps
        self.train_eps = train_eps
        self.build_layers()

    # def build_input_layer(self):
    #     return GINEEdgeConv(
    #         in_channels=self.inp_dim,
    #         out_channels=self.out_dim,
    #         eps=self.eps,
    #         train_eps=self.train_eps,
    #         aggr="add"
    #     )
    def build_input_layer(self):
        # edge_dim = getattr(self, 'edge_dim', None)  # 从图数据中获取边特征维度
        return GINEEdgeConv(
            in_channels=self.inp_dim,
            out_channels=self.out_dim,
            edge_dim=self.out_dim,  # 传递边特征维度
            eps=self.eps,
            train_eps=self.train_eps,
            aggr="add"
        )

    def build_hidden_layer(self):
        # In standard GIN, all layers have same input/output dim (or you can change it)
        # Here we assume fixed out_dim across layers (common in many implementations)
        return GINEEdgeConv(
            in_channels=self.out_dim,
            out_channels=self.out_dim,
            eps=self.eps,
            train_eps=self.train_eps,
            aggr="add"
        )

    def build_message_from_input(self, g):
        return {
            "g": g.edge_index,
            "h": g.x,
            "e": getattr(g, "edge_type", None),      # unused in GIN, but passed
            "he": getattr(g, "edge_attr", None),     # edge features (required)
        }

    def build_message_from_output(self, g, h):
        return {
            "g": g.edge_index,
            "h": h,
            "e": getattr(g, "edge_type", None),
            "he": getattr(g, "edge_attr", None),
        }

    def layer_forward(self, layer, message):
        return self.conv[layer](
            x=message["h"],
            xe=message["he"],
            edge_index=message["g"],
            edge_type=message["e"]  # ignored internally, but passed for interface
        )
    
class SingleHeadAtt(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.sqrt_dim = torch.sqrt(torch.tensor(dim))
        self.Wk = torch.nn.Parameter(torch.zeros((dim, dim)))
        torch.nn.init.xavier_uniform_(self.Wk)
        self.Wq = torch.nn.Parameter(torch.zeros((dim, dim)))
        torch.nn.init.xavier_uniform_(self.Wq)

    def forward(self, key, query, value):
        score = torch.bmm(query, key.transpose(1, 2)) / self.sqrt_dim
        attn = torch.nn.functional.softmax(score, -1)
        context = torch.bmm(attn, value)
        return context, attn


#非单层提取，适用于APPNP
class BinGraphModel(torch.nn.Module):
    def __init__(self, model, llm_name, outdim, task_dim, add_rwpe=None, dropout=0.0, **kwargs):
        super().__init__()
        assert llm_name in LLM_DIM.keys()
        self.model = model
        self.llm_name = llm_name
        self.outdim = outdim
        self.llm_proj = nn.Linear(LLM_DIM[llm_name], outdim)
        self.mlp = MLP([outdim, 2 * outdim, outdim, task_dim], dropout=0.0)
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
        g.edge_attr = self.llm_proj(g.edge_attr)
        return g

    def forward(self, g):
        g = self.initial_projection(g)

        if self.rwpe is not None:
            with torch.no_grad():
                rwpe_norm = self.rwpe_normalization(g.rwpe)
                g.x = torch.cat([g.x, rwpe_norm], dim=-1)
                g.edge_attr = torch.cat(
                    [
                        g.edge_attr,
                        self.edge_rwpe_prior.repeat(len(g.edge_attr), 1),
                    ],
                    dim=-1,
                )
        emb = self.model(g)
        class_emb = emb[g.true_nodes_mask]
        res = self.mlp(class_emb)
        return res

    def freeze_gnn_parameters(self):
        for p in self.model.parameters():
           p.requires_grad = False
        for p in self.mlp.parameters():
            p.requires_grad = False
        for p in self.llm_proj.parameters():
            p.requires_grad = False



class BinGraphAttModel(torch.nn.Module):
    """
    GNN model that use a single layer attention to pool final node representation across
    layers.
    """
    def __init__(self, model, llm_name, outdim, task_dim, add_rwpe=None, dropout=0.0, **kwargs):
        super().__init__()
        assert llm_name in LLM_DIM.keys()
        self.model = model
        self.llm_name = llm_name
        self.outdim = outdim
        self.llm_proj = nn.Linear(LLM_DIM[llm_name], outdim)
        self.mlp = MLP([outdim, 2 * outdim, outdim, task_dim], dropout=0.0)
        self.att = SingleHeadAtt(outdim)
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
        g.edge_attr = self.llm_proj(g.edge_attr)
        return g

    def forward(self, g):
        g = self.initial_projection(g)
        if self.rwpe is not None:
            with torch.no_grad():
                rwpe_norm = self.rwpe_normalization(g.rwpe)
                g.x = torch.cat([g.x, rwpe_norm], dim=-1)
                g.edge_attr = torch.cat(
                    [
                        g.edge_attr,
                        self.edge_rwpe_prior.repeat(len(g.edge_attr), 1),
                    ],
                    dim=-1,
                )
        emb = torch.stack(self.model(g), dim=1)
        query = g.x.unsqueeze(1)
        emb = self.att(emb, query, emb)[0].squeeze()

        class_emb = emb[g.true_nodes_mask]
        res = self.mlp(class_emb)
        return res

    def freeze_gnn_parameters(self):
        for p in self.model.parameters():
           p.requires_grad = False
        for p in self.att.parameters():
            p.requires_grad = False
        for p in self.mlp.parameters():
            p.requires_grad = False
        for p in self.llm_proj.parameters():
            p.requires_grad = False

def mean_pooling(token_embeddings, attention_mask):
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-10)
