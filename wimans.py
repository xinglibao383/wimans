import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd


class WiMANS(Dataset):
    def __init__(self, root_path):
        self.data_parent_path = os.path.join(root_path, 'wifi_csi', 'amp')
        self.label_file_path = os.path.join(root_path, 'annotation.csv')
        self.num_users = 6
        self.num_locations = 5
        self.num_activities = 9
        self.location_map = {
            'a': 0,
            'b': 1,
            'c': 2,
            'd': 3,
            'e': 4,
        }
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
        self.data_filenames = [f for f in os.listdir(self.data_parent_path) if f.endswith('.npy')]
        self.labels = pd.read_csv(self.label_file_path)

    def __len__(self):
        return len(self.data_filenames)

    def __getitem__(self, idx):
        data_filename = self.data_filenames[idx]

        data = np.load(os.path.join(self.data_parent_path, data_filename))
        data_len = data.shape[0]
        if data_len < 3000:
            data = np.pad(data, ((0, 3000 - data_len), (0, 0), (0, 0), (0, 0)), mode='constant', constant_values=0)

        row_id = os.path.splitext(data_filename)[0]
        identity_label, location_label, activity_label = self.get_labels_from_csv(row_id)

        data = torch.tensor(data, dtype=torch.float32)
        identity_label = torch.tensor(identity_label, dtype=torch.float32)
        location_label = torch.tensor(location_label, dtype=torch.float32)
        activity_label = torch.tensor(activity_label, dtype=torch.float32)

        return data, identity_label, location_label, activity_label

    def get_labels_from_csv(self, row_id):
        row = self.labels[self.labels['label'] == row_id]
        if row.empty:
            return [], [], []

        identity_label = np.zeros(self.num_users)
        location_label = np.zeros((self.num_users, self.num_locations))
        activity_label = np.zeros((self.num_users, self.num_activities))

        for i in range(1, 7):
            user_location = row[f'user_{i}_location'].values[0]
            user_activity = row[f'user_{i}_activity'].values[0]

            if user_location and user_activity:
                identity_label[i - 1] = 1

                location_idx = self.location_map.get(user_location, None)
                if location_idx is not None:
                    location_label[i - 1, location_idx] = 1

                activity_idx = self.activity_map.get(user_activity, None)
                if activity_idx is not None:
                    activity_label[i - 1, activity_idx] = 1

        return identity_label, location_label, activity_label


def get_dataloaders(dataset, batch_size, train_ratio=0.7, eval_ratio=0.1):
    total_size = len(dataset)
    train_size = int(total_size * train_ratio)
    eval_size = int(total_size * eval_ratio)
    test_size = total_size - train_size - eval_size

    train_dataset, eval_dataset, test_dataset = random_split(dataset, [train_size, eval_size, test_size])

    return (DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
            DataLoader(eval_dataset, batch_size=batch_size, shuffle=False),
            DataLoader(test_dataset, batch_size=batch_size, shuffle=False))


if __name__ == "__main__":
    dataset = WiMANS(root_path=r'E:\WorkSpace\WiMANS\dataset')
    train_loader, eval_loader, test_loader = get_dataloaders(dataset, batch_size=32)

    for batch_idx, (data, identity_label, location_label, activity_label) in enumerate(train_loader):
        print(f"Train Batch {batch_idx + 1}")
        print("Data Shape:", data.shape)
        print("Identity Label Shape:", identity_label.shape)
        print("Location Label Shape:", location_label.shape)
        print("Activity Label Shape:", activity_label.shape)
        print("-------------------------------------------------")

    for batch_idx, (data, identity_label, location_label, activity_label) in enumerate(eval_loader):
        print(f"Train Batch {batch_idx + 1}")
        print("Data Shape:", data.shape)
        print("Identity Label Shape:", identity_label.shape)
        print("Location Label Shape:", location_label.shape)
        print("Activity Label Shape:", activity_label.shape)
        print("-------------------------------------------------")

    for batch_idx, (data, identity_label, location_label, activity_label) in enumerate(test_loader):
        print(f"Train Batch {batch_idx + 1}")
        print("Data Shape:", data.shape)
        print("Identity Label Shape:", identity_label.shape)
        print("Location Label Shape:", location_label.shape)
        print("Activity Label Shape:", activity_label.shape)
        print("-------------------------------------------------")
