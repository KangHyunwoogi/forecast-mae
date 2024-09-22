from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers.agent_embedding import AgentEmbeddingLayer
from .layers.lane_embedding import LaneEmbeddingLayer
from .layers.multimodal_decoder import MultimodalDecoder
from .layers.vocabulary_decoder import VocabularyDecoder
from .layers.transformer_blocks import Block

import numpy as np
import pandas as pd

class ModelForecast(nn.Module):
    def __init__(
        self,
        embed_dim=128,
        encoder_depth=4,
        num_heads=8,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop_path=0.2,
        future_steps: int = 60,
    ) -> None:
        super().__init__()
        self.hist_embed = AgentEmbeddingLayer(
            4, embed_dim // 4, drop_path_rate=drop_path
        )
        self.lane_embed = LaneEmbeddingLayer(3, embed_dim)

        self.pos_embed = nn.Sequential(
            nn.Linear(4, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

        dpr = [x.item() for x in torch.linspace(0, drop_path, encoder_depth)]
        self.blocks = nn.ModuleList(
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop_path=dpr[i],
            )
            for i in range(encoder_depth)
        )
        self.norm = nn.LayerNorm(embed_dim)

        self.actor_type_embed = nn.Parameter(torch.Tensor(4, embed_dim))
        self.lane_type_embed = nn.Parameter(torch.Tensor(1, 1, embed_dim))

        self.decoder = MultimodalDecoder(embed_dim, future_steps)
        self.vocabulary_decoder = VocabularyDecoder(embed_dim, future_steps)
        self.dense_predictor = nn.Sequential(
            nn.Linear(embed_dim, 256), nn.ReLU(), nn.Linear(256, future_steps * 2)
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # GPU 사용 여부 확인
        # csv_file_path = '/home/ailab/Desktop/wook/forecast-mae/tetest1_2024-09-02-22-13-12.csv'
        csv_file_path = '/home/ailab/Desktop/wook/forecast-mae/argoverse_vocabulary_2048.csv'

        self.vocabulary_trajectories, self.trajectory_candidate_number = self.load_trajectories_from_csv(csv_file_path)
        self.encoded_trajectories = torch.stack([self.encode_trajectory(trajectory) for trajectory in self.vocabulary_trajectories])

        self.initialize_weights()

    def initialize_weights(self):
        nn.init.normal_(self.actor_type_embed, std=0.02)
        nn.init.normal_(self.lane_type_embed, std=0.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def load_from_checkpoint(self, ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu")["state_dict"]
        state_dict = {
            k[len("net.") :]: v for k, v in ckpt.items() if k.startswith("net.")
        }
        return self.load_state_dict(state_dict=state_dict, strict=False)

    def positional_encoding(self, pos, L=32):
        i = torch.arange(L, dtype=torch.float32).unsqueeze(0)  # (1, L)
        pos = pos.unsqueeze(-1)  # (2, 1)

        # Broadcasting pos and i to compute the positional encoding
        gamma_pos = torch.cat([torch.cos(pos / (10000 ** (2 * np.pi * i / L))),
                            torch.sin(pos / (10000 ** (2 * np.pi * i / L)))], dim=-1)
        return gamma_pos.flatten()  # Flatten to a 1D vector

    def encode_trajectory(self, trajectory, L=32):
        # trajectory is expected to be of shape (60, 2)
        
        # Apply positional encoding to each (x, y) pair in the trajectory
        encoded_positions = torch.stack([self.positional_encoding(pos, L) for pos in trajectory], dim=0)
        # Aggregate encoded positions to form a single query vector
        query_vector = encoded_positions.mean(dim=0)  # (L * 2,) -> (256,)
        
        return query_vector

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
        # trajectories_tensor = trajectories_tensor[:50, :, :]  # for debugging
        
        # indices = torch.tensor([178, 291, 101,  13, 376, 228, 106, 252, 186,  22, 150,  27, 285, 321, 131, 171,
        #                 5, 179, 121, 377,  99, 310,  66, 204,  37, 209, 334, 235,  16, 141, 323,  36, 
        #                 14, 203, 256, 262,  24, 103, 343, 270, 343, 234, 322, 207, 109, 263, 153, 38, 142])
        
        # # # 선택된 랜덤 인덱스를 사용해 100개의 trajectory 선택
        # trajectories_tensor = trajectories_tensor[indices, :, :]
        
        return trajectories_tensor, len(trajectories_tensor)

    def forward(self, data):
        hist_padding_mask = data["x_padding_mask"][:, :, :50]
        hist_key_padding_mask = data["x_key_padding_mask"]
        hist_feat = torch.cat(
            [
                data["x"],
                data["x_velocity_diff"][..., None],
                ~hist_padding_mask[..., None],
            ],
            dim=-1,
        )

        B, N, L, D = hist_feat.shape
        hist_feat = hist_feat.view(B * N, L, D)
        hist_feat_key_padding = hist_key_padding_mask.view(B * N)
        actor_feat = self.hist_embed(
            hist_feat[~hist_feat_key_padding].permute(0, 2, 1).contiguous()
        )
        actor_feat_tmp = torch.zeros(
            B * N, actor_feat.shape[-1], device=actor_feat.device
        )
        actor_feat_tmp[~hist_feat_key_padding] = actor_feat
        actor_feat = actor_feat_tmp.view(B, N, actor_feat.shape[-1])

        lane_padding_mask = data["lane_padding_mask"]
        lane_normalized = data["lane_positions"] - data["lane_centers"].unsqueeze(-2)
        lane_normalized = torch.cat(
            [lane_normalized, ~lane_padding_mask[..., None]], dim=-1
        )
        B, M, L, D = lane_normalized.shape
        lane_feat = self.lane_embed(lane_normalized.view(-1, L, D).contiguous())
        lane_feat = lane_feat.view(B, M, -1)

        x_centers = torch.cat([data["x_centers"], data["lane_centers"]], dim=1)
        angles = torch.cat([data["x_angles"][:, :, 49], data["lane_angles"]], dim=1)
        x_angles = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)
        pos_feat = torch.cat([x_centers, x_angles], dim=-1)
        pos_embed = self.pos_embed(pos_feat)

        actor_type_embed = self.actor_type_embed[data["x_attr"][..., 2].long()]
        lane_type_embed = self.lane_type_embed.repeat(B, M, 1)
        actor_feat += actor_type_embed
        lane_feat += lane_type_embed

        x_encoder = torch.cat([actor_feat, lane_feat], dim=1)
        key_padding_mask = torch.cat(
            [data["x_key_padding_mask"], data["lane_key_padding_mask"]], dim=1
        )

        x_encoder = x_encoder + pos_embed
        for blk in self.blocks:
            x_encoder = blk(x_encoder, key_padding_mask=key_padding_mask)
        x_encoder = self.norm(x_encoder)

        # print("x_encoder")
        # print(x_encoder)
        # print(x_encoder.shape)

        # 기존 코드
        # x_agent = x_encoder[:, 0]
        # y_hat, pi = self.decoder(x_agent)

        # x_others = x_encoder[:, 1:N]
        # y_hat_others = self.dense_predictor(x_others).view(B, -1, 60, 2)

        # Use the vocabulary as the query, and encoder output as key and value
        query = self.encoded_trajectories.to(self.device)
        query = query.unsqueeze(0)
        query = query.expand(B, -1, -1)
        x_agent = x_encoder[:, 0, :].unsqueeze(1)  # (B, 1, embed_dim)
        key = value = x_agent  # Use the entire encoder output as key and value
        # print("key")
        # print(key)
        # print("value")
        # print(value)
        # print(value)
        # Pass through the decoder
        y_hat = self.vocabulary_decoder(query, key, value)
        # print(y_hat.shape)
        # print("y_hat")
        # print(y_hat)
        return {
            "y_hat": y_hat,
            # "pi": pi,
            # "y_hat_others": y_hat_others,
        }
