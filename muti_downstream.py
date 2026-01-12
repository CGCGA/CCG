import argparse
import os
from types import SimpleNamespace

from models.MutiGNNmodel import MLPMultiGNNModel
from models.pygGNN import GCN, GIN, APPNPModel
import torch
from pytorch_lightning.loggers import WandbLogger
from torchmetrics import AUROC, Accuracy

import utils
from gp.lightning.data_template import DataModule
from gp.lightning.metric import (
    flat_binary_func,
    EvalKit,
)
from gp.lightning.module_template import ExpConfig
from gp.lightning.training import down_lightning_fit, lightning_fit, up_lightning_fit
from gp.utils.utils import (
    load_yaml,
    combine_dict,
    merge_mod,
    setup_exp,
    set_random_seed,
)
from lightning_model import GraphPredLightning
from models.model import BinGraphModel, BinGraphAttModel
from models.model import PyGRGCNEdge
from task_constructor import UnifiedTaskConstructor
from utils import (
    SentenceEncoder,
    MultiApr,
    MultiAuc,
)

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
    parser.add_argument("--dlr", type=float, default=1e-4)
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

def init_evaluate_kit(text_dataset):
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
    return metrics, val_state, test_state

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
    #将任务字符串转化为任务索引
    val_task_index_lst, val_pool_mode = tasks.construct_exp()

    # 节省GPU空间
    # remove llm model
    if encoder is not None:
        encoder.flush_model()

    """
    2. Construct datasets and lightning datamodule.
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

    wandb_logger = WandbLogger(
        project=params.log_project,
        name=f"params.exp_name",
        save_dir=params.exp_dir,
        offline=params.offline_log,
    )


    # 主循环：一次只处理一个数据集
    for idx, (task_name, d_multiple, d_min_ratio) in enumerate(zip(task_names, data_multiple, min_ratio)):
        print(f"Processing dataset {idx+1}/{len(task_names)}: {task_name}")

        set_random_seed(params.seed)
        
        try:
            # 只传入当前任务的名称
            current_tasks = UnifiedTaskConstructor(
                [task_name],  # 只处理当前任务
                params.load_texts,
                encoder,
                task_config_lookup,
                data_config_lookup,
                batch_size=params.batch_size,
                sample_size=params.train_sample_size,
            )
            
            # 将任务字符串转化为任务索引
            val_task_index_lst, val_pool_mode = current_tasks.construct_exp()

            # 构造当前任务的数据集
            train_data = current_tasks.make_train_data(
                [d_multiple],
                [d_min_ratio], 
                data_val_index=val_task_index_lst
            )

            text_dataset = current_tasks.make_full_dm_list(
                [d_multiple], 
                [d_min_ratio], 
                train_data
            )

            # 4. 创建数据模块
            params.datamodule = DataModule(
                text_dataset, 
                gpu_size=gpu_size, 
                num_workers=params.num_workers
            )

            
            """
            3. Load model 
            """
            #限制输出维度
            in_dim = params.emb_dim + (params.rwpe if params.rwpe is not None else 0)
            out_dim = params.emb_dim + (params.rwpe if params.rwpe is not None else 0)

            gnn1 = GCN(
                params.num_layers,
                in_dim,
                in_dim,
                out_dim,
                dropout=params.dropout,
                JK=params.JK,
            )
            gnn2 = APPNPModel(
                in_dim,
                in_dim,
                out_dim,
                dropout=params.dropout,
                JK=params.JK,
            )
            gnn3 = GIN(
                params.num_layers,
                in_dim,
                in_dim,
                out_dim,
                dropout=params.dropout,
                JK=params.JK,
            )

            muti_model = [gnn1, gnn2, gnn3]
            muti_model = MLPMultiGNNModel(muti_model, llm_name=params.llm_name, outdim=out_dim, task_dim=1,
                              JK=params.JK, add_rwpe=params.rwpe, dropout=params.dropout)

            """
            4. Initiate evaluation kit. 
            """
            metrics, val_state, test_state = init_evaluate_kit(text_dataset=text_dataset)

            """
            5. Initiate optimizer, scheduler and lightning model module.
            """
            optimizer = torch.optim.Adam(
                muti_model.parameters(), lr=params.lr, weight_decay=params.l2
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

            # pred_model = GraphPredLightning(exp_config, muti_model, metrics)
            # checkpoint = torch.load("/home/chenguanwen/OneForAll/checkpoints/model_upstream_muti/in-0/epoch=41-step=457126.ckpt", map_location="cpu")
            # pred_model.load_state_dict(checkpoint['state_dict'])

            pred_model = GraphPredLightning.load_from_checkpoint(
                "./checkpoints/model_multi_upstream_seed1/in-0/epoch=44-step=447223.ckpt",
                exp_config=exp_config,  # 传递必要的参数
                model=muti_model,
                eval_kit=metrics
            )

            strategy = "deepspeed_stage_2" if gpu_size > 1 else "auto"

            # training upstream
            down_lightning_fit(
                wandb_logger,
                pred_model,
                params.datamodule,
                metrics,
                params.num_epochs,
                strategy=strategy,
                save_model=True,
                load_best=params.load_best,
                reload_freq=1,
                test_rep=params.test_rep,
                val_interval=params.val_interval
            )


            
            
        except Exception as e:
            print(f"Error processing dataset {task_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue  # 下一个数据集
        
        finally:
            # 每次循环后释放资源
            if hasattr(current_tasks, 'clear_cache'):
                current_tasks.clear_cache()
            
            # 显式删除大对象
            if 'train_data' in locals():
                del train_data
            if 'text_dataset' in locals():
                del text_dataset
            if 'current_tasks' in locals():
                del current_tasks
            
            # 强制垃圾回收
            import gc
            gc.collect()
            
            # 清理GPU缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    

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
