import os
from tqdm import tqdm
import numpy as np
import torch
import time
import torch.optim as optim
import scipy.io as sio
from scipy.interpolate import interp1d

# MATLAB .mat 파일 불러오기
# train_dataset = "/home/ailab/AILabDataset/03_Shared_Repository/hyunwook/05_Projects/HMG_route_validation/train_dataset_1.mat"
raw_data = sio.loadmat("/home/ailab/AILabDataset/03_Shared_Repository/hyunwook/05_Projects/HMG_route_validation/Transformer/train_dataset_1.mat")

# # MATLAB 구조체 데이터를 파이썬으로 변환
# HmgInterfaceRosbagData = raw_data['HmgInterfaceRosbagData']

# labeled_data = raw_data['olabeledGtData']

surrounding_past_trajectory = raw_data['surrounding_past_trajectory_cell']
surrounding_past_trajectory_temp = surrounding_past_trajectory[2222, 0]

ego_vehicle_velocity = raw_data['ego_velocity_ms']

ego_past_trajectory_x = raw_data['ego_past_trajectory_x']
ego_past_trajectory_y = raw_data['ego_past_trajectory_y']

ego_past_trajectory_x_temp = ego_past_trajectory_x[24]

x = torch.zeros((5, 30, 2))
x_velocity = torch.zeros((6, 30))
x[0, :, 0] = torch.tensor(ego_past_trajectory_x_temp)
x[0, :, 1] = torch.tensor(ego_past_trajectory_y[24])

x_velocity[0, :] = torch.tensor(ego_vehicle_velocity[24])

padding_mask = (x == 0).all(dim=-1)
padding_mask[0,-1] = False

x[:, 1:30] = torch.where(
    (padding_mask[:, :29] | padding_mask[:, 1:30]).unsqueeze(-1),
    torch.zeros(5, 29, 2),
    x[:, 1:30] - x[:, :29],
)
x[:, 0] = torch.zeros(5, 2)

# print(ego_past_trajectory_x_temp)
print(padding_mask)
print(x)
# print(int(surrounding_past_trajectory_temp.shape[0]/2))



# print(surrounding_past_trajectory_temp.shape)

