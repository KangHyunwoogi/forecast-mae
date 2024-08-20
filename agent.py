import os
from tqdm import tqdm
import numpy as np
import torch
import time
import torch.optim as optim
import scipy.io as sio
from scipy.interpolate import interp1d

# MATLAB .mat 파일 불러오기
train_dataset = "/home/ailab/AILabDataset/03_Shared_Repository/hyunwook/05_Projects/HMG_route_validation/train_dataset_1.mat"
raw_data = sio.loadmat(train_dataset)

HmgInterfaceRosbagData = raw_data['HmgInterfaceRosbagData']

surrounding_past_trajectory = raw_data['surrounding_past_trajectory_cell']

num_elements = surrounding_past_trajectory.size

x = torch.zeros((num_elements, 5, 30, 2))
x_velocity = torch.zeros((num_elements, 5, 30))
x_velocity_diff = torch.zeros((num_elements, 5, 30))
x_heading = torch.zeros((num_elements, 5))
x_attr = torch.zeros((num_elements, 5, 3))
x_scored_agents_mask = torch.ones(num_elements, 5, dtype=torch.bool)
x_padding_mask = torch.zeros(num_elements, 5, 30, dtype=torch.bool)

for i in range(num_elements):
    surrounding_agent = surrounding_past_trajectory[i, 0]
    for j in range(5):
        x[i, j, :, 0] = torch.tensor(surrounding_agent[j * 2])       # x 좌표
        x[i, j, :, 1] = torch.tensor(surrounding_agent[j * 2 + 1])   # y 좌표
        
x_attr = x_attr
x_positions = x[:, :30, :2]
x_ctrs = x[:, :, 29, :2]
x_heading = 0
x_velocity = x_velocity
x_velocity_diff = x_velocity_diff
x_padding_mask = (x == 0).all(dim=-1)

print(x.shape)
print(x_padding_mask)