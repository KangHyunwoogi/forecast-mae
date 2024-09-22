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
import cv2

# Drawer class for visualization
class Drawer():
    def __init__(self):
        self.canvas = np.ones((1200, 1500, 3), dtype=np.uint8) * 255
        self.offset_x = 200
        self.offset_y = 600
        self.zoom_ratio = 10

    def draw_trajectory(self, position_x_local, position_y_local, color=(192, 192, 192)):
        """ 기본 경로 그리기, 색상은 기본값으로 회색 """
        for x, y in zip(position_x_local, position_y_local):
            resize_x = int(x * self.zoom_ratio + self.offset_x)
            resize_y = int(y * self.zoom_ratio + self.offset_y)
            cv2.circle(self.canvas, (resize_x, resize_y), 3, color, -1)

            if x == position_x_local[0] and y == position_y_local[0]:
                prev_x = x
                prev_y = y
                continue
            prev_x = int(prev_x * self.zoom_ratio + self.offset_x)
            prev_y = int(prev_y * self.zoom_ratio + self.offset_y)
            cv2.line(self.canvas, (prev_x, prev_y), (resize_x, resize_y), color, 1)
            prev_x = x
            prev_y = y

    def save_plot(self, path):
        cv2.imwrite(path, self.canvas)

    def clear(self):
        self.canvas = np.ones((1200, 1500, 3), dtype=np.uint8) * 255

def visualize_trajectories(trainer, predicted_trajectory, gt_trajectory, batch_idx, y_hat, correct_predictions):
    drawer = Drawer()
    epoch = trainer.current_epoch

    # # 1. 모든 vocabulary trajectories를 회색으로 그리기
    # for idx in range(trainer.vocabulary_trajectories.size(0)):
    #     trajectory = trainer.vocabulary_trajectories[idx]
    #     position_x_local = trajectory[:, 0].cpu().numpy()
    #     position_y_local = trajectory[:, 1].cpu().numpy()
        
    #     # y_hat의 confidence score는 [batch_idx, idx, 0]에서 가져옴
    #     confidence = y_hat[0, idx].item()  # 첫 번째 배치에서 해당 trajectory의 confidence score
        
    #     # confidence score에 따라 회색에서 검은색으로 색상 조절 (0 = 회색, 1 = 검은색)
    #     intensity = int((1 - confidence) * 192)  # intensity가 0에 가까울수록 검은색, 192에 가까울수록 밝은 회색
    #     color = (intensity, intensity, intensity)  # RGB 값으로 색상 설정
        
    #     drawer.draw_trajectory(position_x_local, position_y_local, color=color)  # 회색

    # y_hat의 0번째 batch의 모든 trajectory에 대한 confidence score 가져오기
    
    # print(y_hat.shape)
    
    y_hat_scores = y_hat[0, :]  # batch의 0번째에서 모든 trajectory의 confidence score
    y_hat_sorted, indices = torch.sort(y_hat_scores, descending=False)  # confidence score 기준으로 내림차순 정렬
    
    # print(y_hat_sorted)
    
    # trajectory 개수
    num_trajectories = trainer.vocabulary_trajectories.size(0)

    # print("correct_predictions")
    # print(correct_predictions)

    # if correct_predictions[0] == 1:
    # 1. 정렬된 순서로 vocabulary trajectories를 그리기
    for sorted_idx, idx in enumerate(indices):
        trajectory = trainer.vocabulary_trajectories[idx]
        position_x_local = trajectory[:, 0].cpu().numpy()
        position_y_local = trajectory[:, 1].cpu().numpy()

        # confidence score는 정렬된 y_hat에서 가져옴
        confidence = y_hat_sorted[sorted_idx].item()

        # confidence 값이 큰 것부터 작은 것까지 일정하게 밝아지도록 scaling (높은 값 = 검은색, 낮은 값 = 회색)
        intensity = int((sorted_idx / num_trajectories) * 192)  # 낮은 sorted_idx는 밝고, 높은 sorted_idx는 어두움
        color = (192-intensity,192-intensity,192-intensity)  # 밝은 색에서 어두운 색으로 변환

        drawer.draw_trajectory(position_x_local, position_y_local, color=color)

    # 2. 실제 경로(gt_trajectory)를 파란색으로 그리기
    gt_position_x_local = gt_trajectory[0, :, 0].detach().cpu().numpy()  # 배치에서 첫 번째 데이터만 시각화
    gt_position_y_local = gt_trajectory[0, :, 1].detach().cpu().numpy()
    drawer.draw_trajectory(gt_position_x_local, gt_position_y_local, color=(255, 0, 0))  # 파란색

    # 3. 예측된 경로(predicted_trajectory)를 빨간색으로 그리기
    pred_position_x_local = predicted_trajectory[0, :, 0].detach().cpu().numpy()  # 배치에서 첫 번째 데이터만 시각화
    pred_position_y_local = predicted_trajectory[0, :, 1].detach().cpu().numpy()
    drawer.draw_trajectory(pred_position_x_local, pred_position_y_local, color=(0, 0, 255))  # 빨간색

    # 이미지 저장
    drawer.save_plot(f'/home/ailab/Desktop/wook/forecast-mae/visualization/argoverse_2048/trajectory_plot_epoch_{epoch}_batch_{batch_idx}.png')
    drawer.clear()

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
        
        self.save_visualization = True
        
        # csv_file_path = '/home/ailab/Desktop/wook/forecast-mae/tetest1_2024-09-02-22-13-12.csv'
        csv_file_path = '/home/ailab/Desktop/wook/forecast-mae/argoverse_vocabulary_2048.csv'
        self.vocabulary_trajectories, self.trajectory_candidate_number = self.load_trajectories_from_csv(csv_file_path)
        
    def compute_trajectory_lengths(self, trajectories_tensor):
        """
        각 경로의 길이를 유클리드 거리로 계산
        :param trajectories_tensor: (num_trajectories, 60, 2) 형태의 경로 텐서
        :return: 각 경로의 길이 리스트와 오름차순 정렬된 인덱스
        """
        # 유클리드 거리를 사용하여 각 경로의 길이 계산
        trajectory_lengths = torch.norm(trajectories_tensor[:, 1:, :] - trajectories_tensor[:, :-1, :], dim=2).sum(dim=1)

        # 길이에 따라 오름차순으로 정렬된 인덱스
        sorted_indices = torch.argsort(trajectory_lengths)

        return trajectory_lengths, sorted_indices
        
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
        # 랜덤한 100개의 trajectory 인덱스 선택
        
        trajectory_lengths, sorted_indices = self.compute_trajectory_lengths(trajectories_tensor)
        # print(sorted_indices)
        # torch.seed()
        # random_indices = torch.randperm(trajectories_tensor.size(0))[:50]
        
        # print(random_indices)

        # 선택된 랜덤 인덱스를 사용해 100개의 trajectory 선택
        # trajectories_tensor = trajectories_tensor[random_indices, :, :]
        
        # indices = torch.tensor([178, 291, 101,  13, 376, 228, 106, 252, 186,  22, 150,  27, 285, 321, 131, 171,
        #                 5, 179, 121, 377,  99, 310,  66, 204,  37, 209, 334, 235,  16, 141, 323,  36, 
        #                 14, 203, 256, 262,  24, 103, 343, 270, 343, 234, 322, 207, 109, 263, 153, 38, 142])
        
        # # # 선택된 랜덤 인덱스를 사용해 100개의 trajectory 선택
        # trajectories_tensor = trajectories_tensor[indices, :, :]
        
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

    def cal_loss(self, out, data, batch_idx):
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
            
            # print("predicted_trajectory_idx")
            # print(predicted_trajectory_idx)
            # print(closest_trajectory_idx.squeeze(-1))
            
            # Compare with the ground truth closest trajectory index (gt_trajectory)
            correct_predictions = (predicted_trajectory_idx == closest_trajectory_idx.squeeze(-1)).float()
            
            # print(correct_predictions.shape)
            
            # Calculate accuracy
            accuracy = correct_predictions.mean()
            
            print(accuracy)
    
        # Log the accuracy for debugging/monitoring
        self.log("train/accuracy", accuracy, on_step=True, on_epoch=True, prog_bar=True)

        reshaped_vocabulary_trajectory = self.vocabulary_trajectories[closest_trajectory_idx].squeeze(1).to(self.device)

        if batch_idx % 50 == 0 and self.save_visualization:
            visualize_trajectories(self, reshaped_vocabulary_trajectory, gt_trajectory, batch_idx, y_hat, correct_predictions)

        return {
            "loss": weighted_classification_loss,
            "Accuracy": accuracy,
            # "classification_loss": weighted_classification_loss.item(),
            # "l2_loss": l2_loss.item(),
            # "loss": loss,
            # "reg_loss": agent_reg_loss.item(),
            # "cls_loss": agent_cls_loss.item(),
        }

    def training_step(self, data, batch_idx):
        out = self(data)
        losses = self.cal_loss(out, data, batch_idx)

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
        losses = self.cal_loss(out, data, batch_idx)
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
