import os.path as osp
from typing import Optional, Union, List

import torch
from lightning.pytorch import LightningModule

from utils.config import ExpConfig
from utils.evaluate import EvalKit

# class MutiBaseTemplate(LightningModule):
#     def __init__(
#         self,
#         exp_config: ExpConfig,
#         model: torch.nn.Module,
#         eval_kit: Optional[EvalKit] = None,
#         name: str = "",
#     ):

#         super().__init__()

#         self.exp_config = exp_config

#         self.model = model
#         self.name = name

#         self.eval_kit = eval_kit

#     def on_test_epoch_start(self):
#         self.on_validation_epoch_start()

#     def configure_optimizers(self):
#         optimizer = self.exp_config.get_optimizer()
#         optimizer_dict = {"optimizer": optimizer}
#         if self.exp_config.get_scheduler() is not None:
#             optimizer_dict["lr_scheduler"] = self.exp_config.get_scheduler()
#         return optimizer_dict
    
#     def compute_results(
#             self, batch, batch_idx, step_name, log_loss=True, *args
#         ):
#         """
#         Args:
#             outputs: [tensor1, tensor2, tensor3] - 每个模型的输出
#             batch: 批次数据
#         """

#         try:
#             # score = self(batch, *args)
#             # weight_scores = []
#             # for i,s in enumerate(score):
#             #     weight_score = self.model.weights[i] * s
#             #     weight_scores.append(weight_score)

#             # score = sum(weight_scores)
#             # loss = self.eval_kit.compute_loss(score, batch)

#             score = self.model(batch)  # 调用EnsembleMultiGNNModel的forward，返回集成输出
#             # 不需要再加权，因为模型内部已经完成加权
#             loss = self.eval_kit.compute_loss(score, batch)

#             if self.model.weights.grad is not None:
#                 print(f"Weights grad norm: {self.model.weights.grad.norm()}")

#         except RuntimeError as e:
#             if "out of memory" in str(e):
#                 print("Ignoring OOM batch")
#                 loss = None
#                 score = None
#             else:
#                 raise
#         if loss is not None:
#             self.log(
#                 osp.join(self.name, step_name, "loss"),
#                 loss,
#                 on_step=False,
#                 on_epoch=True,
#                 prog_bar=log_loss,
#                 batch_size=batch.batch_size
#                 if hasattr(batch, "batch_size")
#                 else len(batch),
#             )
#         with torch.no_grad():
#             if self.eval_kit.has_eval_state(step_name):
#                 self.eval_kit.eval_step(score, batch, step_name)
#         return score, loss

#     def epoch_post_process(self, epoch_name):
#         if self.eval_kit.has_eval_state(epoch_name):
#             metric = self.eval_kit.eval_epoch(epoch_name)
#             self.log(
#                 self.eval_kit.get_metric_name(epoch_name),
#                 metric,
#                 prog_bar=True,
#                 sync_dist=True
#             )
#             self.eval_kit.eval_reset(epoch_name)
#             return metric

#     def training_step(self, batch, batch_idx, dataloader_idx=0):
#         score, loss = self.compute_results(
#             batch, batch_idx, self.exp_config.train_state_name[dataloader_idx]
#         )
#         return loss

#     def on_train_epoch_end(self):
#         for name in self.exp_config.train_state_name:
#             self.epoch_post_process(name)

#     def validation_step(self, batch, batch_idx, dataloader_idx=0):
#         self.compute_results(
#             batch,
#             batch_idx,
#             self.exp_config.val_state_name[dataloader_idx],
#             log_loss=False,
#         )

#     def on_validation_epoch_end(self):
#         cur_metric = []
#         for name in self.exp_config.val_state_name:
#             metric = self.epoch_post_process(name)
#             if metric is not None:
#                 cur_metric.append(metric.cpu())
#         if self.exp_config.dataset_callback is not None:
#             self.exp_config.dataset_callback(cur_metric)

#     def test_step(self, batch, batch_idx, dataloader_idx=0):
#         self.compute_results(
#             batch,
#             batch_idx,
#             self.exp_config.test_state_name[dataloader_idx],
#             log_loss=False,
#         )

#     def on_test_epoch_end(self):
#         for name in self.exp_config.test_state_name:
#             self.epoch_post_process(name)


class MutiBaseTemplate(LightningModule):
    def __init__(
        self,
        exp_config: ExpConfig,
        model: torch.nn.Module,
        eval_kit: Optional[EvalKit] = None,
        name: str = "",
    ):

        super().__init__()

        self.exp_config = exp_config

        self.model = model
        self.name = name

        self.eval_kit = eval_kit

    def on_test_epoch_start(self):
        self.on_validation_epoch_start()

    def configure_optimizers(self):
        # 确保优化器包含所有需要训练的参数，包括集成权重
        optimizer = self.exp_config.get_optimizer()
        
        # 检查模型中是否有需要梯度的参数
        trainable_params = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                trainable_params.append(param)
                # print(f"Training parameter: {name}, requires_grad: {param.requires_grad}")
        
        print(f"Total trainable parameters: {len(trainable_params)}")
        
        # 如果使用自定义优化器，确保它包含模型的所有参数
        if hasattr(optimizer, 'param_groups'):
            optimizer.param_groups[0]['params'] = trainable_params
        
        optimizer_dict = {"optimizer": optimizer}
        if self.exp_config.get_scheduler() is not None:
            optimizer_dict["lr_scheduler"] = self.exp_config.get_scheduler()
        return optimizer_dict
    
    def compute_results(
            self, batch, batch_idx, step_name, log_loss=True, *args
        ):
        """
        Args:
            outputs: [tensor1, tensor2, tensor3] - 每个模型的输出
            batch: 批次数据
        """

        try:
            # 直接调用模型的forward方法，它内部已经完成了加权聚合
            score = self.model(batch)  # 调用EnsembleMultiGNNModel的forward，返回集成输出
            loss = self.eval_kit.compute_loss(score, batch)

            # # 检查权重梯度
            # # 损失计算后，权重的梯度会被计算
            # if self.model.raw_weights.grad is not None:
            #     print(f"Weights grad norm: {self.model.raw_weights.grad.norm()}")

        except RuntimeError as e:
            if "out of memory" in str(e):
                print("Ignoring OOM batch")
                loss = None
                score = None
            else:
                raise
        
        if loss is not None:
            self.log(
                osp.join(self.name, step_name, "loss"),
                loss,
                on_step=False,
                on_epoch=True,
                prog_bar=log_loss,
                batch_size=batch.batch_size
                if hasattr(batch, "batch_size")
                else len(batch),
            )
        with torch.no_grad():
            if self.eval_kit.has_eval_state(step_name):
                self.eval_kit.eval_step(score, batch, step_name)
        return score, loss

    def epoch_post_process(self, epoch_name):
        if self.eval_kit.has_eval_state(epoch_name):
            metric = self.eval_kit.eval_epoch(epoch_name)
            self.log(
                self.eval_kit.get_metric_name(epoch_name),
                metric,
                prog_bar=True,
                sync_dist=True
            )
            self.eval_kit.eval_reset(epoch_name)
            return metric

    def training_step(self, batch, batch_idx, dataloader_idx=0):
        score, loss = self.compute_results(
            batch, batch_idx, self.exp_config.train_state_name[dataloader_idx]
        )
        return loss

    def on_train_epoch_end(self):
        for name in self.exp_config.train_state_name:
            self.epoch_post_process(name)

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        self.compute_results(
            batch,
            batch_idx,
            self.exp_config.val_state_name[dataloader_idx],
            log_loss=False,
        )

    def on_validation_epoch_end(self):
        cur_metric = []
        for name in self.exp_config.val_state_name:
            metric = self.epoch_post_process(name)
            if metric is not None:
                cur_metric.append(metric.cpu())
        if self.exp_config.dataset_callback is not None:
            self.exp_config.dataset_callback(cur_metric)

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        self.compute_results(
            batch,
            batch_idx,
            self.exp_config.test_state_name[dataloader_idx],
            log_loss=False,
        )

    def on_test_epoch_end(self):
        for name in self.exp_config.test_state_name:
            self.epoch_post_process(name)

class MutiGraphPredLightning(MutiBaseTemplate):
    def forward(self, batch):
        return self.model(batch)