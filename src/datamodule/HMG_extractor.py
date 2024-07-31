import os
from tqdm import tqdm
import numpy as np
import torch
import time
import torch.optim as optim
import scipy.io as sio
from scipy.interpolate import interp1d

class HMGExtractor:
    def __init__(
        self,
        radius: float = 150,
        mode: str = "train",
        remove_outlier_actors: bool = True,
    ) -> None:
        self.mode = mode
        self.radius = radius
        self.remove_outlier_actors = remove_outlier_actors

    def save(self, file: str):
        pass
    
    def get_data(self, raw_data, index):
        return self.process(raw_data, index)
    
    def process(self, raw_data, index):
        HmgInterfaceRosbagData = raw_data['HmgInterfaceRosbagData']
        surrounding_past_trajectory = raw_data['surrounding_past_trajectory_cell']
        labeled_data = raw_data['olabeledGtData']
        labeled_data_torch = torch.from_numpy(labeled_data)
        y = labeled_data_torch[index]
        
        percept_lane_info = HmgInterfaceRosbagData['RtmFrCmrInfo_t'][0, 0]
        hdmap_lane_info = HmgInterfaceRosbagData['PathSet_t'][0, 0]  # 1x8000 구조체 배열 추출
        # num_elements = hdmap_lane_info.size

        num_points = 30
        
        percept_lane_left_x = np.zeros((num_points))
        percept_lane_right_x = np.zeros((num_points))
        percept_lane_left_y = np.zeros((num_points))
        percept_lane_right_y = np.zeros((num_points))
        left_view_range = np.zeros((2))
        right_view_range = np.zeros((2))
                
        # for i in range(num_elements):
            # 각 요소의 값을 추출하여 저장
        lane_model_param = percept_lane_info[0]['LaneModelParam_t'][index]
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

        left_view_range[0] = left_range_start
        left_view_range[1] = left_range_end

        right_view_range[0] = right_range_start
        right_view_range[1] = right_range_end

        # x 범위 설정
        x_left = np.linspace(left_range_start, left_range_end, num_points)
        x_right = np.linspace(right_range_start, right_range_end, num_points)
        # 3차 방정식 계산: y = left_x0 + left_x1*x + left_x2*x^2 + left_x3*x^3
        y_left = left_x0 + left_x1 * x_left + left_x2 * x_left**2 + left_x3 * x_left**3
        y_right = right_x0 + right_x1 * x_right + right_x2 * x_right**2 + right_x3 * x_right**3

        # 결과 저장
        percept_lane_left_x[:] = x_left
        percept_lane_right_x[:] = x_right
        percept_lane_left_y[:] = y_left
        percept_lane_right_y[:] = y_right
                
        map_lane_left_y = np.zeros((num_points))
        map_lane_right_y = np.zeros((num_points))

        # for i in range(num_elements):
            # 각 요소의 값을 추출하여 저장
        path_info_t = hdmap_lane_info[0]['PathInfo_t'][index]  # 1x1 struct 접근
        route_info = path_info_t['RouteInfo'][0, 0]  # 1x1 struct 접근
        
        left_x = route_info['f64LeftRouteLon'][0,0]
        left_y = route_info['f64LeftRouteLat'][0,0]
        right_x = route_info['f64RightRouteLon'][0,0]
        right_y = route_info['f64RightRouteLat'][0,0]
        
        left_range_start = left_view_range[0]
        left_range_end = left_view_range[1]
        left_valid_indices = (left_x > left_range_start - 3) & (left_x < left_range_end) & (left_y < 9) & (left_y > -9)

        right_range_start = right_view_range[0]
        right_range_end = right_view_range[1]
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
        map_lane_left_y[:len(filtered_interpolated_left_y)] = filtered_interpolated_left_y
        map_lane_right_y[:len(filtered_interpolated_right_y)] = filtered_interpolated_right_y

        # ego_past_trajectory_x = raw_data['ego_past_trajectory_x']
        # ego_past_trajectory_y = raw_data['ego_past_trajectory_y']

        # for i in range(num_elements):
        surrounding_past_trajectory_temp = surrounding_past_trajectory[index, 0]
        surrounding_past_trajectory_data = surrounding_past_trajectory_temp.reshape(5,2,30)
        surrounding_past_trajectory_data = surrounding_past_trajectory_data.transpose(0,2,1)
            
            
        # lane preprocess
        max_points_per_segments = 30

        percept_lane_center_x = (percept_lane_left_x + percept_lane_right_x) / 2
        percept_lane_center_y = (percept_lane_left_y + percept_lane_right_y) / 2

        perception_lane_segments, perception_lane_attr, perception_is_intersection = [], [], []

        # 반복문 외부에서 초기화
        perception_lane_segment = np.zeros((max_points_per_segments, 2))

        # for i in range(num_elements):
            # 반복문 내부에서는 각각의 레인 세그먼트를 텐서로 변환
        perception_lane_segment[:, 0] = percept_lane_center_x[:]
        perception_lane_segment[:, 1] = percept_lane_center_y[:]
        perception_lane_segments.append(torch.from_numpy(perception_lane_segment).float())  # numpy 배열을 torch.Tensor로 변환
        perception_attribute = torch.tensor([0, 0, 0], dtype=torch.float)
        perception_lane_attr.append(perception_attribute)
        perception_is_intersection.append(0)
            
        perception_lane_positions = torch.stack(perception_lane_segments, dim=0)
        # perception_lane_positions = perception_lane_positions.unsqueeze(1)
        perception_lane_attr = torch.stack(perception_lane_attr, dim=0)
        # perception_lane_attr = perception_lane_attr.unsqueeze(1)
        perception_is_intersections = torch.Tensor(perception_is_intersection)
        # perception_is_intersections = perception_is_intersections.unsqueeze(1)
        perception_lane_ctrs = perception_lane_positions[:, 9:11].mean(dim=1)
        # perception_lane_ctrs = perception_lane_ctrs.unsqueeze(1)
        perception_lane_angles = torch.atan2(
                    perception_lane_positions[:, 10, 1] - perception_lane_positions[:, 9, 1],
                    perception_lane_positions[:, 10, 0] - perception_lane_positions[:, 9, 0],
                )
        # perception_lane_angles = perception_lane_angles.unsqueeze(1)
        perception_lane_padding_mask = (
            (perception_lane_positions[:, :, 0] > 1000)
            | (perception_lane_positions[:, :, 0] < -1000)
            | (perception_lane_positions[:, :, 1] > 1000)
            | (perception_lane_positions[:, :, 1] < -1000)
        )

        # print(perception_lane_positions.size())
        # print(perception_lane_attr.size())
        # print(perception_is_intersections.size())
        # print(perception_lane_ctrs.size())
        # print(perception_lane_angles.size())
        # print(perception_lane_padding_mask.size())


        ########################################################################
        lane_segments, lane_attr, is_intersection = [], [], []

        lane_center_x = (percept_lane_left_x + percept_lane_right_x) / 2
        lane_center_y = (map_lane_left_y + map_lane_right_y) / 2

        lane_segment = np.zeros((max_points_per_segments, 2))

        # for i in range(num_elements):
            # 반복문 내부에서는 각각의 레인 세그먼트를 텐서로 변환
        lane_segment[:, 0] = lane_center_x[:]
        lane_segment[:, 1] = lane_center_y[:]
        lane_segments.append(torch.from_numpy(lane_segment).float())  # numpy 배열을 torch.Tensor로 변환
        attribute = torch.tensor([0, 0, 0], dtype=torch.float)
        lane_attr.append(attribute)
        is_intersection.append(0)
            
        lane_positions = torch.stack(lane_segments, dim=0)
        # lane_positions = lane_positions.unsqueeze(1)
        lane_attr = torch.stack(lane_attr, dim=0)
        # lane_attr = lane_attr.unsqueeze(1)
        is_intersections = torch.Tensor(is_intersection)
        # is_intersections = is_intersections.unsqueeze(1)
        lane_ctrs = lane_positions[:, 9:11].mean(dim=1)
        # lane_ctrs = lane_ctrs.unsqueeze(1)
        lane_angles = torch.atan2(
                    lane_positions[:, 10, 1] - lane_positions[:, 9, 1],
                    lane_positions[:, 10, 0] - lane_positions[:, 9, 0],
                )
        # perception_lane_angles = perception_lane_angles.unsqueeze(1)
        lane_padding_mask = (
            (lane_positions[:, :, 0] > 1000)
            | (lane_positions[:, :, 0] < -1000)
            | (lane_positions[:, :, 1] > 1000)
            | (lane_positions[:, :, 1] < -1000)
        )
        
        x = torch.zeros((5, 30, 2))
        x_velocity = torch.zeros((5, 30))
        x_velocity_diff = torch.zeros((5, 30))
        x_heading = torch.zeros((5, 30))
        x_attr = torch.zeros((5, 3))
        x_scored_agents_mask = torch.ones(5, dtype=torch.bool)
        x_padding_mask = torch.zeros(5, 30, dtype=torch.bool)

        # for i in range(num_elements):
        surrounding_agent = surrounding_past_trajectory[index, 0]
        for j in range(5):
            x[j, :, 0] = torch.tensor(surrounding_agent[j * 2])       # x 좌표
            x[j, :, 1] = torch.tensor(surrounding_agent[j * 2 + 1])   # y 좌표
                
        x_attr = x_attr
        x_positions = x[:, :30, :2]
        x_ctrs = x[:, 29, :2]
        # x_heading = 0
        x_velocity = x_velocity
        x_velocity_diff = x_velocity_diff
        padding_mask = (x == 0).all(dim=-1)
        # GT 값 넣어야 함 Need to be fixed
        # y = None
        
        origin = torch.tensor([0, 0], dtype=torch.float)
        theta = torch.tensor([0], dtype=torch.float)
        scenario_id = torch.tensor([0], dtype=torch.int) 
        agent_id = torch.tensor([0], dtype=torch.int) 
        city = torch.tensor([0], dtype=torch.int) 
        
        return {
            "x": x[:, :50],
            "y": y,
            "x_attr": x_attr,
            "x_positions": x_positions,
            "x_centers": x_ctrs,
            "x_angles": x_heading,
            "x_velocity": x_velocity,
            "x_velocity_diff": x_velocity_diff,
            "x_padding_mask": padding_mask,
            "lane_positions": lane_positions,
            "lane_centers": lane_ctrs,
            "lane_angles": lane_angles,
            "lane_attr": lane_attr,
            "lane_padding_mask": lane_padding_mask,
            "perception_lane_positions": perception_lane_positions,
            "perception_lane_centers": perception_lane_ctrs,
            "perception_lane_angles": perception_lane_angles,
            "perception_lane_attr": perception_lane_attr,
            "perception_lane_padding_mask": perception_lane_padding_mask,
            "is_intersections": is_intersections,
            "origin": origin.view(-1, 2),
            "theta": theta,
            "scenario_id": scenario_id,
            "track_id": agent_id,
            "city": city,
        }
        pass