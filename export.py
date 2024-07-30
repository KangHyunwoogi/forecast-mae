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

# MATLAB 구조체 데이터를 파이썬으로 변환
HmgInterfaceRosbagData = raw_data['HmgInterfaceRosbagData']

# HmgInterfaceRosbagData['PathSet_t']의 구조 확인
percept_lane_info = HmgInterfaceRosbagData['RtmFrCmrInfo_t'][0, 0]
hdmap_lane_info = HmgInterfaceRosbagData['PathSet_t'][0, 0]  # 1x8000 구조체 배열 추출
num_elements = hdmap_lane_info.size

num_points = 30

percept_lane_left_x = np.zeros((num_elements, num_points))
percept_lane_right_x = np.zeros((num_elements, num_points))
percept_lane_left_y = np.zeros((num_elements, num_points))
percept_lane_right_y = np.zeros((num_elements, num_points))
left_view_range = np.zeros((num_elements, 2))
right_view_range = np.zeros((num_elements, 2))

for i in range(num_elements):
    # 각 요소의 값을 추출하여 저장
    lane_model_param = percept_lane_info[0]['LaneModelParam_t'][i]
    ego_left_lane = lane_model_param['EgoLeftLane'][0, 0]
    ego_right_lane = lane_model_param['EgoRightLane'][0, 0]
    
    left_x0 = -1 * ego_left_lane['f64LanePosition'][0,0].item()
    left_x1 = -1 * ego_left_lane['f64LaneHeadingAngle'][0,0].item()
    left_x2 = -1 * ego_left_lane['f64LaneCurvature'][0,0].item()
    left_x3 = -1 * ego_left_lane['f64LaneCurvRat'][0,0].item()
    left_range_start = ego_left_lane['f64RangeStart'][0,0].item()
    left_range_end = ego_left_lane['f64RangeEnd'][0,0].item()

    right_x0 = -1 * ego_right_lane['f64LanePosition'][0, 0].item()
    right_x1 = -1 * ego_right_lane['f64LaneHeadingAngle'][0, 0].item()
    right_x2 = -1 * ego_right_lane['f64LaneCurvature'][0, 0].item()
    right_x3 = -1 * ego_right_lane['f64LaneCurvRat'][0, 0].item()
    right_range_start = ego_right_lane['f64RangeStart'][0, 0].item()
    right_range_end = ego_right_lane['f64RangeEnd'][0, 0].item()

    left_view_range[i, 0] = left_range_start
    left_view_range[i, 1] = left_range_end

    right_view_range[i, 0] = right_range_start
    right_view_range[i, 1] = right_range_end

    # x 범위 설정
    x_left = np.linspace(left_range_start, left_range_end, num_points)
    x_right = np.linspace(right_range_start, right_range_end, num_points)
    # 3차 방정식 계산: y = left_x0 + left_x1*x + left_x2*x^2 + left_x3*x^3
    y_left = left_x0 + left_x1 * x_left + left_x2 * x_left**2 + left_x3 * x_left**3
    y_right = right_x0 + right_x1 * x_right + right_x2 * x_right**2 + right_x3 * x_right**3

    # 결과 저장
    percept_lane_left_x[i, :] = x_left
    percept_lane_right_x[i, :] = x_right
    percept_lane_left_y[i, :] = y_left
    percept_lane_right_y[i, :] = y_right

# print(percept_lane_left_y)

# 배열 초기화
map_lane_left_y = np.zeros((num_elements, num_points))
map_lane_right_y = np.zeros((num_elements, num_points))

for i in range(num_elements):
    # 각 요소의 값을 추출하여 저장
    path_info_t = hdmap_lane_info[0]['PathInfo_t'][i]  # 1x1 struct 접근
    route_info = path_info_t['RouteInfo'][0, 0]  # 1x1 struct 접근
    
    left_x = route_info['f64LeftRouteLon'][0,0]
    left_y = route_info['f64LeftRouteLat'][0,0]
    right_x = route_info['f64RightRouteLon'][0,0]
    right_y = route_info['f64RightRouteLat'][0,0]
    
    left_range_start = left_view_range[i, 0]
    left_range_end = left_view_range[i, 1]
    left_valid_indices = (left_x > left_range_start - 3) & (left_x < left_range_end) & (left_y < 9) & (left_y > -9)

    right_range_start = right_view_range[i, 0]
    right_range_end = right_view_range[i, 1]
    right_valid_indices = (right_x > right_range_start - 3) & (right_x < right_range_end) & (right_y < 9) & (right_y > -9)

    filtered_left_y = left_y[left_valid_indices]
    filtered_right_y = right_y[right_valid_indices]

    # y값을 cubic spline을 사용하여 보간
    if len(filtered_left_y) > 1:
        f_left = interp1d(np.arange(len(filtered_left_y)), filtered_left_y, kind='cubic', fill_value="extrapolate")
        filtered_interpolated_left_y = f_left(np.linspace(0, len(filtered_left_y) - 1, num_points))
    else:
        filtered_interpolated_left_y = np.zeros(num_points)

    if len(filtered_right_y) > 1:
        f_right = interp1d(np.arange(len(filtered_right_y)), filtered_right_y, kind='cubic', fill_value="extrapolate")
        filtered_interpolated_right_y = f_right(np.linspace(0, len(filtered_right_y) - 1, num_points))
    else:
        filtered_interpolated_right_y = np.zeros(num_points)

    # 각 행에 값을 할당
    map_lane_left_y[i, :len(filtered_interpolated_left_y)] = filtered_interpolated_left_y
    map_lane_right_y[i, :len(filtered_interpolated_right_y)] = filtered_interpolated_right_y

# 결과 배열: map_lane_left_y, map_lane_right_y

print(map_lane_left_y)

ego_past_trajectory_x = raw_data['ego_past_trajectory_x']
ego_past_trajectory_y = raw_data['ego_past_trajectory_y']

surrounding_past_trajectory = raw_data['surrounding_past_trajectory_cell']

for i in range(num_elements):
    surrounding_past_trajectory_temp = surrounding_past_trajectory[i, 0]
    surrounding_past_trajectory_data = surrounding_past_trajectory_temp.reshape(5,2,30)
    surrounding_past_trajectory_data = surrounding_past_trajectory_data.transpose(0,2,1)
    
    
# lane preprocess

max_points_per_segments = 30

lane_segments, lane_attr, is_intersection = [], [], []

percept_lane_center_x = (percept_lane_left_x + percept_lane_right_x) / 2
percept_lane_center_y = (percept_lane_left_y + percept_lane_right_y) / 2

# 반복문 외부에서 초기화
lane_segment = np.zeros((max_points_per_segments, 2))

for i in range(num_elements):
    # 반복문 내부에서는 각각의 레인 세그먼트를 텐서로 변환
    lane_segment[:, 0] = percept_lane_center_x[i, :]
    lane_segment[:, 1] = percept_lane_center_y[i, :]
    lane_segments.append(torch.from_numpy(lane_segment).float())  # numpy 배열을 torch.Tensor로 변환
    lane_attr.append([0, 0, 0])
    is_intersection.append(False)
    
lane_positions = torch.stack(lane_segments, dim=0)
lane_attr = torch.stack(lane_attr, dim=0)
is_intersections = torch.Tensor(is_intersection)

print(lane_positions.size())

    