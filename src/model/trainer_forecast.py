import datetime
from pathlib import Path

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import MetricCollection

from src.metrics import MR, minADE, minFDE
from src.utils.optim import WarmupCosLR
from src.utils.submission_av2 import SubmissionAv2

from .model_forecast import ModelForecast

import numpy as np
import pandas as pd

import math

class Trainer(pl.LightningModule):
    def __init__(
        self,
        dim=128,
        historical_steps=50,
        future_steps=60,
        encoder_depth=4,
        num_heads=8,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop_path=0.2,
        pretrained_weights: str = None,
        lr: float = 1e-3,
        warmup_epochs: int = 10,
        epochs: int = 60,
        weight_decay: float = 1e-4,
    ) -> None:
        super(Trainer, self).__init__()
        self.warmup_epochs = warmup_epochs
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.save_hyperparameters()
        self.submission_handler = SubmissionAv2()

        self.net = ModelForecast(
            embed_dim=dim,
            encoder_depth=encoder_depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop_path=drop_path,
            future_steps=future_steps,
        )

        if pretrained_weights is not None:
            self.net.load_from_checkpoint(pretrained_weights)

        metrics = MetricCollection(
            {
                # "minADE1": minADE(k=1),
                # "minADE6": minADE(k=6),
                # "minFDE1": minFDE(k=1),
                # "minFDE6": minFDE(k=6),
                # "MR": MR(),
            }
        )
        self.val_metrics = metrics.clone(prefix="val_")
        
        csv_file_path = '/home/ailab/Desktop/wook/forecast-mae/trajectory_data.csv'
        self.vocabulary_trajectories, self.trajectory_candidate_number = self.load_trajectories_from_csv(csv_file_path)
        
    def load_trajectories_from_csv(self, file_path):
        # CSV 파일 읽기
        df = pd.read_csv(file_path)

        # path_id별로 그룹화
        grouped = df.groupby('path_id')

        # 각 path_id에 대해 60x2 형태로 변환된 데이터 저장
        trajectories = []

        for _, group in grouped:
            # time 순으로 정렬
            group = group.sort_values(by='time')

            # x, y 좌표를 가져와 numpy 배열로 변환
            xy = group[['x', 'y']].values

            # 만약 60 timesteps보다 적다면 패딩(예: 0으로) 추가
            if len(xy) < 60:
                padding = np.zeros((60 - len(xy), 2))
                xy = np.vstack((xy, padding))
            elif len(xy) > 60:
                xy = xy[:60]  # 60 timesteps로 자르기

            trajectories.append(xy)
        # 결과를 tensor로 변환 (path_id, 60, 2) 형태
        trajectories_tensor = torch.tensor(trajectories, dtype=torch.float32)
        trajectories_tensor = trajectories_tensor[:50, :, :]  # for debugging
        
        return trajectories_tensor, len(trajectories_tensor)

    def forward(self, data):
        return self.net(data)

    def predict(self, data):
        with torch.no_grad():
            out = self.net(data)
        predictions, prob = self.submission_handler.format_data(
            data, out["y_hat"], out["pi"], inference=True
        )
        return predictions, prob

    def cal_loss(self, out, data):
        # y_hat, pi = out["y_hat"], out["pi"]
        y_hat = out["y_hat"]
        gt_trajectory = data["y"][:, 0]  # Assume the ground truth trajectory is in 'data["y"][:, 0]'

        vocabulary_distances = torch.norm(self.vocabulary_trajectories.unsqueeze(0).to(self.device) - gt_trajectory.unsqueeze(1), dim=-1).sum(dim=-1)  # (B, vocabulary_size)\
        closest_trajectory_idx = torch.argmin(vocabulary_distances, dim=-1)  # (B,)
        closest_trajectory_idx = closest_trajectory_idx.unsqueeze(-1)  # (B, 1)
        
        # print("closest_trajectory_idx")
        # print(closest_trajectory_idx)
        
        target = F.one_hot(closest_trajectory_idx, num_classes=self.vocabulary_trajectories.size(0)).float()
        
        # print
        # print("vocabulary_distances")
        # print(vocabulary_distances.shape)
        # print(vocabulary_distances)
        
        # Reshape y_hat and target to match the expected shape for F.cross_entropy
        y_hat = y_hat.view(-1, self.vocabulary_trajectories.size(0))  # (B, vocabulary_size)
        target = target.view(-1, self.vocabulary_trajectories.size(0))  # (B, vocabulary_size)
        
        # torch.set_printoptions(edgeitems=torch.inf, threshold=torch.inf)
        # print(target)
        
        # print("y_hat")
        # print(y_hat)
        # print("target")
        # print(target)
        
        classification_loss = F.binary_cross_entropy_with_logits(y_hat, target)

        # 1. 추가적인 거리 기반 가중치를 적용한 BCE loss
        gt_distance = torch.norm(gt_trajectory - self.vocabulary_trajectories[closest_trajectory_idx].to(self.device), dim=-1)  # (B,)

        # print(gt_trajectory.shape)
        # print(self.vocabulary_trajectories.shape)
        # print(classification_loss.shape)
        # print(gt_distance.shape)
        
        # 거리가 멀수록 가중치를 더 높게 설정
        # clipped_distance = torch.clamp(gt_distance, max=20)
        selected_vocabulary_distances = torch.gather(vocabulary_distances, 1, closest_trajectory_idx)
        clipped_distance = torch.sqrt(selected_vocabulary_distances)
        distance_weight = F.softplus(clipped_distance)  # 거리 기반 가중치
        # print("distance_weight")
        # print(distance_weight)
        weighted_classification_loss = (classification_loss * distance_weight).mean()
        
        # # 2. L2 거리 손실 추가
        # l2_loss = F.mse_loss(y_hat, gt_trajectory)  # 예측 값과 실제 경로 간의 거리 기반 손실
        
        # # 최종 손실은 거리 기반 가중치를 반영한 분류 손실 + L2 손실
        # total_loss = weighted_classification_loss + l2_loss

        print("weighted_classification_loss")
        print(weighted_classification_loss)
        print("classification_loss")
        print(classification_loss)
        # print("l2_loss")
        # print(l2_loss)
        # print("total_loss")
        # print(total_loss)
        

        # # Calculate L2 distance between predicted trajectory and all vocabulary trajectories
        # distances = torch.norm(y_hat - gt_trajectory.unsqueeze(1), dim=-1).sum(dim=-1)

        # # Find the index of the closest trajectory in the vocabulary to the ground truth trajectory
        # best_mode = torch.argmin(distances, dim=-1)
        # y_hat_best = y_hat[torch.arange(y_hat.shape[0]), best_mode]

        # # Calculate regression loss and classification loss
        # agent_reg_loss = F.smooth_l1_loss(y_hat_best[..., :2], gt_trajectory)
        # # agent_cls_loss = F.cross_entropy(pi, best_mode.detach())

        # Total loss
        # loss = agent_reg_loss # + agent_cls_loss
        # Accuracy Calculation
        with torch.no_grad():
            # Get the predicted trajectory indices from y_hat (max probability indices)
            predicted_trajectory_idx = torch.argmax(y_hat, dim=-1)  # (B,)
            
            # Compare with the ground truth closest trajectory index (gt_trajectory)
            correct_predictions = (predicted_trajectory_idx == closest_trajectory_idx.squeeze(-1)).float()
            
            # print(correct_predictions.shape)
            
            # Calculate accuracy
            accuracy = correct_predictions.mean()
    
        # Log the accuracy for debugging/monitoring
        self.log("train/accuracy", accuracy, on_step=True, on_epoch=True, prog_bar=True)

        return {
            "loss": weighted_classification_loss,
            "Accuracy": accuracy,
            # "classification_loss": weighted_classification_loss.item(),
            # "l2_loss": l2_loss.item(),
            # "loss": loss,
            # "reg_loss": agent_reg_loss.item(),
            # "cls_loss": agent_cls_loss.item(),
        }
        
        # 원본 코드
        # y_hat, pi, y_hat_others = out["y_hat"], out["pi"], out["y_hat_others"]
        # y, y_others = data["y"][:, 0], data["y"][:, 1:]

        # l2_norm = torch.norm(y_hat[..., :2] - y.unsqueeze(1), dim=-1).sum(dim=-1)
        # best_mode = torch.argmin(l2_norm, dim=-1)
        # y_hat_best = y_hat[torch.arange(y_hat.shape[0]), best_mode]

        # agent_reg_loss = F.smooth_l1_loss(y_hat_best[..., :2], y)
        # agent_cls_loss = F.cross_entropy(pi, best_mode.detach())

        # others_reg_mask = ~data["x_padding_mask"][:, 1:, 50:]
        # others_reg_loss = F.smooth_l1_loss(
        #     y_hat_others[others_reg_mask], y_others[others_reg_mask]
        # )

        # loss = agent_reg_loss + agent_cls_loss + others_reg_loss

        # return {
        #     "loss": loss,
        #     "reg_loss": agent_reg_loss.item(),
        #     "cls_loss": agent_cls_loss.item(),
        #     "others_reg_loss": others_reg_loss.item(),
        # }

    def training_step(self, data, batch_idx):
        out = self(data)
        losses = self.cal_loss(out, data)

        for k, v in losses.items():
            self.log(
                f"train/{k}",
                v,
                on_step=True,
                on_epoch=True,
                prog_bar=False,
                sync_dist=True,
            )

        return losses["loss"]

    def validation_step(self, data, batch_idx):
        out = self(data)
        losses = self.cal_loss(out, data)
        metrics = self.val_metrics(out, data["y"][:, 0])

        self.log(
            "val/loss",
            losses["loss"],
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
        )
        self.log(
            "val/Accuracy",
            losses["Accuracy"],
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
        )
        self.log_dict(
            metrics,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=1,
            sync_dist=True,
        )

    def on_test_start(self) -> None:
        save_dir = Path("./submission")
        save_dir.mkdir(exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
        self.submission_handler = SubmissionAv2(
            save_dir=save_dir, filename=f"forecast_mae_{timestamp}"
        )

    def test_step(self, data, batch_idx) -> None:
        out = self(data)
        self.submission_handler.format_data(data, out["y_hat"], out["pi"])

    def on_test_end(self) -> None:
        self.submission_handler.generate_submission_file()

    def configure_optimizers(self):
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (
            nn.Linear,
            nn.Conv1d,
            nn.Conv2d,
            nn.Conv3d,
            nn.MultiheadAttention,
            nn.LSTM,
            nn.GRU,
        )
        blacklist_weight_modules = (
            nn.BatchNorm1d,
            nn.BatchNorm2d,
            nn.BatchNorm3d,
            nn.SyncBatchNorm,
            nn.LayerNorm,
            nn.Embedding,
        )
        for module_name, module in self.named_modules():
            for param_name, param in module.named_parameters():
                full_param_name = (
                    "%s.%s" % (module_name, param_name) if module_name else param_name
                )
                if "bias" in param_name:
                    no_decay.add(full_param_name)
                elif "weight" in param_name:
                    if isinstance(module, whitelist_weight_modules):
                        decay.add(full_param_name)
                    elif isinstance(module, blacklist_weight_modules):
                        no_decay.add(full_param_name)
                elif not ("weight" in param_name or "bias" in param_name):
                    no_decay.add(full_param_name)
        param_dict = {
            param_name: param for param_name, param in self.named_parameters()
        }
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0
        assert len(param_dict.keys() - union_params) == 0

        optim_groups = [
            {
                "params": [
                    param_dict[param_name] for param_name in sorted(list(decay))
                ],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [
                    param_dict[param_name] for param_name in sorted(list(no_decay))
                ],
                "weight_decay": 0.0,
            },
        ]

        optimizer = torch.optim.AdamW(
            optim_groups, lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = WarmupCosLR(
            optimizer=optimizer,
            lr=self.lr,
            min_lr=1e-6,
            warmup_epochs=self.warmup_epochs,
            epochs=self.epochs,
        )
        return [optimizer], [scheduler]
