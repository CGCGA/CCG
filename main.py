import argparse
import os
from types import SimpleNamespace

import torch
from pytorch_lightning.loggers import WandbLogger
from torchmetrics import AUROC, Accuracy

import utils.utils as utils
from utils.data import DataModule
from utils.evaluate import (
    flat_binary_func,
    EvalKit,
)
from utils.config import ExpConfig, GraphPredLightning
from utils.train import lightning_fit
from utils.utils import (
    load_yaml,
    combine_dict,
    merge_mod,
    setup_exp,
    set_random_seed,
    MultiApr,
    MultiAuc,
)
from utils.model import BinGraphModel, BinGraphAttModel, SentenceEncoder
from utils.model import PyGRGCNEdge, PyGAPPNPEdge, PyGGINEEdge
from utils.constructor import UnifiedTaskConstructor
from utils.testonGNN import EdgeEnhancedAPPNP

def parse_args():
    parser = argparse.ArgumentParser()
    # Model parameters
    # parser.add_argument("--lm_type", type=str, default="microsoft/deberta-base")
    parser.add_argument("--lm_type", type=str, default="../m/deberta-v3-base")
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--activation", type=str, default="gelu")
    parser.add_argument("--norm", type=str, default="layernorm")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--negative_slope", type=float, default=0.2)
    
    # Training parameters
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--mask_rate", type=float, default=0.15)
    parser.add_argument("--lam", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.996)
    parser.add_argument("--delayed_ema_epoch", type=int, default=10)
    
    # PPR sampling parameters
    parser.add_argument("--ppr_alpha", type=float, default=0.15)
    parser.add_argument("--ppr_top_k", type=int, default=32)
    
    # Instruction tuning parameters
    parser.add_argument("--llm_name", type=str, default="meta-llama/Llama-2-7b")
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    
    # Other parameters
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    
    return parser.parse_args()

def main(params):
    """
    0. GPU检查
    """
    device, gpu_ids = utils.get_available_devices()
    gpu_size = len(gpu_ids)

    """
    1. 加载编码器
    """
    encoder = SentenceEncoder(params.llm_name, batch_size=params.llm_b_size)


   #数据集任务设置
    task_config_lookup = load_yaml(
        os.path.join(os.path.dirname(__file__), "configs", "task_config.yaml")
    )
    data_config_lookup = load_yaml(os.path.join(os.path.dirname(__file__), "configs", "data_config.yaml"))

   #将task_names转化为列表，若为字符串类型则转换
    if isinstance(params.task_names, str):
        task_names = [a.strip() for a in params.task_names.split(",")]
    else:
        task_names = params.task_names

    tasks = UnifiedTaskConstructor(
        task_names,
        params.load_texts,
        encoder,
        task_config_lookup,
        data_config_lookup,
        batch_size=params.batch_size,
        sample_size=params.train_sample_size,
    )
    #配置方法
    val_task_index_lst, val_pool_mode = tasks.construct_exp()

    # 节省GPU空间
    # remove llm model
    if encoder is not None:
        encoder.flush_model()

    """
    2. Load model 
    """
    #限制输出维度
    out_dim = params.emb_dim + (params.rwpe if params.rwpe is not None else 0)

    gnn1 = PyGRGCNEdge(
        params.num_layers,
        5,
        out_dim,
        out_dim,
        drop_ratio=params.dropout,
        JK=params.JK,
    )
    gnn2 = PyGAPPNPEdge(
        10,
        5,
        out_dim,
        out_dim,
        drop_ratio=params.dropout,
        JK=params.JK,
    )
    gnn3 = PyGGINEEdge(
        params.num_layers,
        0,
        out_dim,
        out_dim,
        drop_ratio=params.dropout,
        JK=params.JK,
    )

    gnn4 = EdgeEnhancedAPPNP(
        in_channels=out_dim,
        hidden_channels=128,
        out_channels=out_dim,
        edge_attr_dim=out_dim,
        K=15,
        alpha=0.1,
        dropout=0.5
    ).to(device)

    #编码器
    bin_model = BinGraphAttModel if params.JK == "none" else BinGraphModel
    model1 = bin_model(model=gnn1, llm_name=params.llm_name, outdim=out_dim, task_dim=1,
                      add_rwpe=params.rwpe, dropout=params.dropout)
    model2 = bin_model(model=gnn2, llm_name=params.llm_name, outdim=out_dim, task_dim=1,
                      add_rwpe=params.rwpe, dropout=params.dropout)
    model3 = bin_model(model=gnn3, llm_name=params.llm_name, outdim=out_dim, task_dim=1,
                      add_rwpe=params.rwpe, dropout=params.dropout)
    
    model4 = bin_model(model=gnn4, llm_name=params.llm_name, outdim=out_dim, task_dim=1,
                      add_rwpe=params.rwpe, dropout=params.dropout)
    

    """
    3. Construct datasets and lightning datamodule.
    """

    #设置两个参数，表明数据增强强度和数据筛选下限阈值
    if hasattr(params, "d_multiple"):
        if isinstance(params.d_multiple, str):
            data_multiple = [float(a) for a in params.d_multiple.split(",")]
        else:
            data_multiple = params.d_multiple
    else:
        data_multiple = [1]

    if hasattr(params, "d_min_ratio"):
        if isinstance(params.d_min_ratio, str):
            min_ratio = [float(a) for a in params.d_min_ratio.split(",")]
        else:
            min_ratio = params.d_min_ratio
    else:
        min_ratio = [1]


    train_data = tasks.make_train_data(data_multiple, min_ratio, data_val_index=val_task_index_lst)

    text_dataset = tasks.make_full_dm_list(
        data_multiple, min_ratio, train_data
    )
    params.datamodule = DataModule(
        text_dataset, gpu_size=gpu_size, num_workers=params.num_workers
    )

    """
    4. Initiate evaluation kit. 
    """
    eval_data = text_dataset["val"] + text_dataset["test"]
    val_state = [dt.state_name for dt in text_dataset["val"]]
    test_state = [dt.state_name for dt in text_dataset["test"]]
    eval_state = val_state + test_state
    eval_metric = [dt.metric for dt in eval_data]
    eval_funcs = [dt.meta_data["eval_func"] for dt in eval_data]
    loss = torch.nn.BCEWithLogitsLoss()
    evlter = []
    for dt in eval_data:
        if dt.metric == "acc":
            evlter.append(Accuracy(task="multiclass", num_classes=dt.classes))
        elif dt.metric == "auc":
            evlter.append(AUROC(task="binary"))
        elif dt.metric == "apr":
            evlter.append(MultiApr(num_labels=dt.classes))
        elif dt.metric == "aucmulti":
            evlter.append(MultiAuc(num_labels=dt.classes))
    metrics = EvalKit(
        eval_metric,
        evlter,
        loss,
        eval_funcs,
        flat_binary_func,
        eval_mode="max",
        exp_prefix="",
        eval_state=eval_state,
        val_monitor_state=val_state[0],
        test_monitor_state=test_state[0],
    )

    """
    5. Initiate optimizer, scheduler and lightning model module.
    """
    optimizer = torch.optim.Adam(
        model4.parameters(), lr=params.lr, weight_decay=params.l2
    )
    lr_scheduler = {
        "scheduler": torch.optim.lr_scheduler.StepLR(optimizer, 15, 0.5),
        "interval": "epoch",
        "frequency": 1,
    }

    exp_config = ExpConfig(
        "",
        optimizer,
        dataset_callback=train_data.update,
        lr_scheduler=lr_scheduler,
    )
    exp_config.val_state_name = val_state
    exp_config.test_state_name = test_state

    pred_model = GraphPredLightning(exp_config, model4, metrics)

    """
    6. Start training and logging.
    """
    wandb_logger = WandbLogger(
        project=params.log_project,
        name=params.exp_name,
        save_dir=params.exp_dir,
        offline=params.offline_log,
    )

    strategy = "deepspeed_stage_2" if gpu_size > 1 else "auto"
    val_res, test_res = lightning_fit(
        wandb_logger,
        pred_model,
        params.datamodule,
        metrics,
        params.num_epochs,
        strategy=strategy,
        save_model=False,
        load_best=params.load_best,
        reload_freq=1,
        test_rep=params.test_rep,
        val_interval=params.val_interval
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="rl")
    parser.add_argument("--override", type=str)

    parser.add_argument(
        "opts",
        default=[],
        nargs=argparse.REMAINDER,
        help="Modify config options using the command-line",
    )

    params = parser.parse_args()
    configs = []
    configs.append(
        load_yaml(
            os.path.join(
                os.path.dirname(__file__), "configs", "default_config.yaml"
            )
        )
    )

    if params.override is not None:
        override_config = load_yaml(params.override)
        configs.append(override_config)
    # Add for few-shot parameters

    mod_params = combine_dict(*configs)
    mod_params = merge_mod(mod_params, params.opts)
    setup_exp(mod_params)

    params = SimpleNamespace(**mod_params)
    set_random_seed(params.seed)

    torch.set_float32_matmul_precision("high")
    params.log_project = "full_cdm"

    params.exp_name += f"_{params.llm_name}_ofa1"

    print(params)
    main(params)
