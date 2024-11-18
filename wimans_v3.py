import os
import math
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd


class WiMANS(Dataset):
    def __init__(self, root_path, nperseg=512, noverlap=128, nfft=1024, window='hamming', remove_static=False):
        self.spectra_parent_path = os.path.join(root_path, 'wifi_csi', 'stft',
                                                f'{nperseg}_{noverlap}_{nfft}_{window}_without_static' if remove_static else f'{nperseg}_{noverlap}_{nfft}_{window}_with_static')
        # phase_unwrapped is better than phase
        # self.csi_parent_path = os.path.join(root_path, 'wifi_csi', 'phase')
        self.csi_parent_path = os.path.join(root_path, 'wifi_csi', 'phase_unwrapped')
        self.label_file_path = os.path.join(root_path, 'annotation.csv')
        self.num_users = 6
        self.num_locations = 5
        self.num_activities = 9
        self.location_map = {
            'a': 1,
            'b': 2,
            'c': 3,
            'd': 4,
            'e': 5,
        }
        self.activity_map = {
            'nothing': 1,
            'walk': 2,
            'rotation': 3,
            'jump': 4,
            'wave': 5,
            'lie_down': 6,
            'pick_up': 7,
            'sit_down': 8,
            'stand_up': 9
        }
        self.data_filenames = [f for f in os.listdir(self.csi_parent_path) if f.endswith('.npy')]
        self.labels = pd.read_csv(self.label_file_path)

    def __len__(self):
        return len(self.data_filenames)

    def __getitem__(self, idx):
        data_filename = self.data_filenames[idx]

        csi = np.load(os.path.join(self.csi_parent_path, data_filename))
        csi_len = csi.shape[0]
        if csi_len < 3000:
            csi = np.pad(csi, ((0, 3000 - csi_len), (0, 0), (0, 0), (0, 0)), mode='constant', constant_values=0)

        row_id = os.path.splitext(data_filename)[0]
        identity_label, location_label, activity_label = self.get_labels_from_csv(row_id)

        spectra = np.load(os.path.join(self.spectra_parent_path, data_filename))

        csi = torch.tensor(csi, dtype=torch.float32)
        spectra = torch.tensor(spectra, dtype=torch.float32)
        identity_label = torch.tensor(identity_label, dtype=torch.long)
        location_label = torch.tensor(location_label, dtype=torch.long)
        activity_label = torch.tensor(activity_label, dtype=torch.long)

        return csi, spectra, identity_label, location_label, activity_label

    def get_labels_from_csv(self, row_id):
        row = self.labels[self.labels['label'] == row_id]
        if row.empty:
            return [], [], []

        identity_label = np.zeros(self.num_users)
        location_label = np.zeros((self.num_users))
        activity_label = np.zeros((self.num_users))

        for i in range(1, 7):
            user_location = row[f'user_{i}_location'].values[0]
            user_activity = row[f'user_{i}_activity'].values[0]

            if user_location and user_activity:
                identity_label[i - 1] = 1

                location_idx = self.location_map.get(user_location, None)
                if location_idx is not None:
                    location_label[i - 1] = location_idx

                activity_idx = self.activity_map.get(user_activity, None)
                if activity_idx is not None:
                    activity_label[i - 1] = activity_idx

        return identity_label, location_label, activity_label


def get_dataloaders_random_split(dataset, batch_size, train_ratio=0.7, eval_ratio=0.1):
    total_size = len(dataset)
    train_size = int(total_size * train_ratio)
    eval_size = int(total_size * eval_ratio)
    test_size = total_size - train_size - eval_size

    train_dataset, eval_dataset, test_dataset = random_split(dataset, [train_size, eval_size, test_size])

    return (DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
            DataLoader(eval_dataset, batch_size=batch_size, shuffle=False),
            DataLoader(test_dataset, batch_size=batch_size, shuffle=False))


def get_dataloaders(dataset, batch_size, train_ratio=0.7, eval_ratio=0.1):
    total_size = len(dataset)

    train_size = int(total_size * train_ratio)
    eval_size = int(total_size * eval_ratio)
    test_size = total_size - train_size - eval_size

    train_indices, eval_indices, test_indices = [], [], []

    train_end = int(train_ratio * 10)
    eval_end = int(train_ratio * 10) + int(eval_ratio * 10)
    test_end = 10

    for i in range(0, total_size, 10):
        train_indices.extend(range(i, i + train_end))
        eval_indices.extend(range(i + train_end, i + eval_end))
        test_indices.extend(range(i + eval_end, i + test_end))

    train_dataset = torch.utils.data.Subset(dataset, train_indices[:train_size])
    eval_dataset = torch.utils.data.Subset(dataset, eval_indices[:eval_size])
    test_dataset = torch.utils.data.Subset(dataset, test_indices[:test_size])

    return (DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
            DataLoader(eval_dataset, batch_size=batch_size, shuffle=False),
            DataLoader(test_dataset, batch_size=batch_size, shuffle=False))


if __name__ == "__main__":
    dataset = WiMANS(root_path=r'E:\WorkSpace\WiMANS\dataset')
    train_loader, eval_loader, test_loader = get_dataloaders(dataset, batch_size=32)

    for batch_idx, (csi, spectra, identity_label, location_label, activity_label) in enumerate(train_loader):
        print(f"Train Batch {batch_idx + 1}")
        print("CSI Shape:", csi.shape)
        print("Spectra Shape:", spectra.shape)
        print("Identity Label Shape:", identity_label.shape)
        print("Location Label Shape:", location_label.shape)
        print("Activity Label Shape:", activity_label.shape)
        print("-------------------------------------------------")

    for batch_idx, (csi, spectra, identity_label, location_label, activity_label) in enumerate(eval_loader):
        print(f"Train Batch {batch_idx + 1}")
        print("CSI Shape:", csi.shape)
        print("Spectra Shape:", spectra.shape)
        print("Identity Label Shape:", identity_label.shape)
        print("Location Label Shape:", location_label.shape)
        print("Activity Label Shape:", activity_label.shape)
        print("-------------------------------------------------")

    for batch_idx, (csi, spectra, identity_label, location_label, activity_label) in enumerate(test_loader):
        print(f"Train Batch {batch_idx + 1}")
        print("CSI Shape:", csi.shape)
        print("Spectra Shape:", spectra.shape)
        print("Identity Label Shape:", identity_label.shape)
        print("Location Label Shape:", location_label.shape)
        print("Activity Label Shape:", activity_label.shape)
        print("-------------------------------------------------")
