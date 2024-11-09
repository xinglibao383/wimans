import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from typing import List, Tuple


class WiMANS(Dataset):
    def __init__(self, npy_dir: str, csv_file: str, target_length: int = 3000):
        """
        初始化 WiMANS 数据集

        参数:
        npy_dir: str - .npy 文件所在的目录路径
        csv_file: str - CSV 文件路径，包含活动标签信息
        target_length: int - 目标时间步数，默认3000
        """
        self.npy_dir = npy_dir
        self.csv_file = csv_file
        self.target_length = target_length
        self.activity_map = {
            'nothing': 0,
            'walk': 1,
            'rotation': 2,
            'jump': 3,
            'wave': 4,
            'lie_down': 5,
            'pick_up': 6,
            'sit_down': 7,
            'stand_up': 8
        }
        self.npy_files = [f for f in os.listdir(npy_dir) if f.endswith('.npy')]


    def __len__(self):
        """返回数据集的大小"""
        return len(self.npy_files)

    def get_labels_from_csv(self, activity_id):
        """
        从CSV文件中获取指定活动ID的标签

        参数:
        activity_id: str - 活动标签ID

        返回:
        user_activity_status: 用户是否参与活动的状态列表
        user_activity_type: 用户活动类型的 one-hot 编码列表
        """
        # 读取CSV文件
        df = pd.read_csv(self.csv_file)

        # 查找活动标签对应的行
        activity_row = df[df['label'] == activity_id]

        # 如果找不到对应的行，返回空的列表
        if activity_row.empty:
            return [], []

        # 初始化用户活动状态和活动类型的列表
        user_activity_status = []
        user_activity_type = []

        # 遍历6个用户，获取活动状态和活动类型
        for i in range(1, 7):
            user_activity_column = f'user_{i}_activity'

            # 获取该用户的活动信息
            user_activity = activity_row[user_activity_column].values[0]

            # 判断该用户是否参与活动（如果活动列为空，说明该用户没有活动）
            user_in_activity = not pd.isna(user_activity)

            # 将状态列表中的值设为 [0, 1]（在活动中）或 [1, 0]（不在活动中）
            if user_in_activity:
                user_activity_status.append([0, 1])  # 用户参与活动
            else:
                user_activity_status.append([1, 0])  # 用户不参与活动

            # 如果用户参与活动，则根据活动编号返回one-hot向量
            if user_in_activity:
                activity_num = self.activity_map.get(user_activity, -1)
                # 创建一个长度为9的one-hot向量
                one_hot = [0] * 9
                if activity_num != -1:
                    one_hot[activity_num] = 1
                user_activity_type.append(one_hot)
            else:
                # 如果没有活动，返回一个全零的one-hot向量
                user_activity_type.append([0] * 9)

        return user_activity_status, user_activity_type

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """获取指定索引的数据和标签"""
        # 获取 .npy 文件路径
        npy_file_name = self.npy_files[idx]
        npy_file_path = os.path.join(self.npy_dir, npy_file_name)

        # 读取 .npy 文件
        npy_data = np.load(npy_file_path)

        # 检查数据的长度是否少于目标长度
        current_length = npy_data.shape[0]
        if current_length < self.target_length:
            # 如果数据不足3000，填充到3000
            padding = self.target_length - current_length
            npy_data = np.pad(npy_data, ((0, padding), (0, 0), (0, 0), (0, 0)), mode='constant', constant_values=0)

        # 获取文件名（去除后缀）作为活动标签的ID
        activity_id = os.path.splitext(npy_file_name)[0]

        # 调用 get_labels_from_csv 函数来获取标签
        user_activity_status, user_activity_type = self.get_labels_from_csv(activity_id)

        # 转换为Tensor
        npy_data_tensor = torch.tensor(npy_data, dtype=torch.float32)
        user_activity_status_tensor = torch.tensor(user_activity_status, dtype=torch.float32)
        user_activity_type_tensor = torch.tensor(user_activity_type, dtype=torch.float32)

        # 返回数据（.npy 文件内容）、用户活动状态和活动类型
        return npy_data_tensor, user_activity_status_tensor, user_activity_type_tensor


if __name__ == "__main__":
    # 使用 DataLoader 加载数据
    npy_dir = r'E:\WorkSpace\WiMANS\dataset\wifi_csi\amp'  # .npy 文件所在的目录
    csv_file = r'E:\WorkSpace\WiMANS\dataset\annotation.csv'  # CSV 文件路径

    # 创建数据集对象
    dataset = WiMANS(npy_dir, csv_file)

    # 创建DataLoader对象
    data_loader = DataLoader(dataset, batch_size=8, shuffle=True)

    # 测试数据加载
    for batch_idx, (npy_data, user_activity_status, user_activity_type) in enumerate(data_loader):
        print(f"Batch {batch_idx + 1}")
        print("Numpy Data Shape:", npy_data.shape)
        print("User Activity Status:", user_activity_status.shape)
        print("User Activity Type:", user_activity_type.shape)
        print("-------------------------------------------------")
