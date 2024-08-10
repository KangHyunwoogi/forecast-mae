from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

# from .av2_extractor import Av2Extractor
from .HMG_extractor import HMGExtractor

import scipy.io as sio
import os

class Av2Dataset(Dataset):
    def __init__(
        self,
        data_root: Path,
        cached_split: str = None,
        # extractor: Av2Extractor = None,
        # extractor : HMGExtractor = None,
        extractor = HMGExtractor(),
        data_file : str = None,
    ):
        super(Av2Dataset, self).__init__()
        self.data = self.load_data(data_root, data_file)

        if cached_split is not None:
            self.data_folder = Path(data_root) / cached_split
            self.file_list = sorted(list(self.data_folder.glob("*.pt")))
            self.load = True
        elif extractor is not None:
            self.extractor = extractor
            self.data_folder = Path(data_root)
            print(f"Extracting data from {self.data_folder}")
            self.file_list = list(self.data_folder.rglob("*.parquet"))
            self.load = False
        elif data_file is not None:
            self.data_folder = Path(data_root)
            print(f"Extracting data from {self.data_folder}")
            self.extractor = extractor
            self.load = False
        else:
            raise ValueError("Either cached_split or extractor must be specified")

        # print(
        #     f"data root: {data_root}, total number of files: {len(self.file_list)}"
        # )
        
        print(f"data root: {data_root}")

    def __len__(self) -> int:
        HmgInterfaceRosbagData = self.data['HmgInterfaceRosbagData']
        hdmap_lane_info = HmgInterfaceRosbagData['PathSet_t'][0, 0]
        return hdmap_lane_info.size

    def __getitem__(self, index: int):
        if self.load:
            data = torch.load(self.file_list[index])
        else:
            data = self.extractor.get_data(self.data, index)

        return data

    def load_data(self, data_root, data_file):
        data_root = Path(data_root)
        
        file_path = data_root / data_file        
        mat_data = sio.loadmat(file_path)
        return mat_data

def collate_fn(batch):
    data = {}

    for key in [
        "x",
        "x_attr",
        "x_positions",
        "x_centers",
        "x_angles",
        "x_velocity",
        "x_velocity_diff",
        "lane_positions",
        "lane_centers",
        "lane_angles",
        "lane_attr",
        "is_intersections",
        "perception_lane_positions",
        "perception_lane_centers",
        "perception_lane_angles",
        "perception_lane_attr",
        
    ]:
        data[key] = pad_sequence([b[key] for b in batch], batch_first=True)

    if "x_scored" in batch[0]:
        data["x_scored"] = pad_sequence(
            [b["x_scored"] for b in batch], batch_first=True
        )

    if batch[0]["y"] is not None:
        data["y"] = pad_sequence([b["y"] for b in batch], batch_first=True)

    for key in ["x_padding_mask", "lane_padding_mask", "perception_lane_padding_mask"]:
        data[key] = pad_sequence(
            [b[key] for b in batch], batch_first=True, padding_value=True
        )

    data["x_key_padding_mask"] = data["x_padding_mask"].all(-1)
    data["lane_key_padding_mask"] = data["lane_padding_mask"].all(-1)
    data["perception_lane_key_padding_mask"] = data["perception_lane_padding_mask"].all(-1)
    data["num_actors"] = (~data["x_key_padding_mask"]).sum(-1)
    data["num_lanes"] = (~data["lane_key_padding_mask"]).sum(-1)
    data["perception_num_lanes"] = (~data["perception_lane_key_padding_mask"]).sum(-1)

    data["scenario_id"] = [b["scenario_id"] for b in batch]
    data["track_id"] = [b["track_id"] for b in batch]

    data["origin"] = torch.cat([b["origin"] for b in batch], dim=0)
    data["theta"] = torch.cat([b["theta"] for b in batch])

    return data
