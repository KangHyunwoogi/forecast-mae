import pandas as pd
import torch
import numpy as np

def load_trajectories_from_csv(file_path):
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
    return trajectories_tensor

# 사용 예시
file_path = '/home/ailab/Desktop/wook/forecast-mae/trajectory_data.csv'
trajectories_tensor = load_trajectories_from_csv(file_path)

print(trajectories_tensor.shape)