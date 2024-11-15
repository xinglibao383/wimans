import os
import numpy as np
import pandas as pd
from scipy.signal import stft


class STFTBatchComputer:
    def __init__(self, root_path):
        self.root_path = root_path
        self.mapping = {
            'classroom_2.4': 'act_201_1',
            'classroom_5.0': 'act_202_1',
            'meeting_room_2.4': 'act_203_1',
            'meeting_room_5.0': 'act_204_1',
            'empty_room_2.4': 'act_205_1',
            'empty_room_5.0': 'act_206_1'
        }
        self.df = pd.read_csv(os.path.join(self.root_path, 'annotation.csv'))

    def get_static_filepath(self, filepath):
        label_value, _ = os.path.splitext(os.path.basename(filepath))
        row = self.df[self.df['label'] == label_value]

        environment = row['environment'].values[0]
        wifi_band = row['wifi_band'].values[0]

        return os.path.join(self.root_path, 'wifi_csi', 'amp', f"{self.mapping.get(f'{environment}_{wifi_band}')}.npy")

    def remove_static_component(self, filepath):
        data, static_data = np.load(filepath), np.load(self.get_static_filepath(filepath))
        data_time, static_data_time = data.shape[0], static_data.shape[0]
        # [time, transmitter, receiver, subcarrier] -> [transmitter * receiver * subcarrier, time]
        data, static_data = data.reshape(data_time, 270).T, static_data.reshape(static_data_time, 270).T

        subcarrier_means = np.mean(static_data, axis=1, keepdims=True)
        data = data - subcarrier_means

        data, static_data = data.T, static_data.T
        data, static_data = data.reshape(data_time, 3, 3, 30), static_data.reshape(static_data_time, 3, 3, 30)

        return data, static_data

    def compute_stft(self, filepath, nperseg=512, noverlap=128, nfft=1024, window='hamming', fs=1000,
                     remove_static=False):
        if remove_static:
            csi_data, _ = self.remove_static_component(filepath)
        else:
            csi_data = np.load(filepath)

        time, transmitter_count, receiver_count, subcarrier_count = csi_data.shape
        if time < 3000:
            csi_data = np.pad(csi_data, ((0, 3000 - time), (0, 0), (0, 0), (0, 0)), mode='constant', constant_values=0)

        Zxx_270 = []

        for tx in range(transmitter_count):
            for rx in range(receiver_count):
                for sc in range(subcarrier_count):
                    signal = csi_data[:, tx, rx, sc]
                    freqs, _, Zxx = stft(signal, fs=fs, nperseg=nperseg, noverlap=noverlap, nfft=nfft, window=window)
                    freq_range = np.logical_and(freqs >= 0, freqs <= 60)
                    Zxx_270.append(np.abs(Zxx[freq_range, :]))  # 使用绝对值保留幅度信息

        return np.array(Zxx_270)

    def compute_save_all_stft(self, nperseg=512, noverlap=128, nfft=1024, window='hamming', fs=1000,
                              remove_static=False):
        parent_path = os.path.join(self.root_path, 'wifi_csi', 'amp')
        save_parent_path = os.path.join(self.root_path, 'wifi_csi', 'stft')
        if remove_static:
            save_parent_path = os.path.join(save_parent_path, f'{nperseg}_{noverlap}_{nfft}_{window}_without_static')
        else:
            save_parent_path = os.path.join(save_parent_path, f'{nperseg}_{noverlap}_{nfft}_{window}_with_static')
        if not os.path.exists(save_parent_path):
            os.makedirs(save_parent_path)

        for filename in os.listdir(parent_path):
            if filename.endswith('.npy'):
                Zxx_270 = self.compute_stft(os.path.join(parent_path, filename),
                                            nperseg=nperseg, noverlap=noverlap, nfft=nfft, window=window, fs=fs,
                                            remove_static=remove_static)
                save_path = os.path.join(save_parent_path, filename)
                np.save(save_path, Zxx_270)
                print(f"Saved STFT result for {filename} to {save_path}")


if __name__ == "__main__":
    batchComputer = STFTBatchComputer(root_path=r'E:\WorkSpace\WiMANS\dataset')

    batchComputer.compute_save_all_stft(nperseg=512, noverlap=128, nfft=1024, window='hamming', fs=1000,
                                        remove_static=False)
    batchComputer.compute_save_all_stft(nperseg=512, noverlap=128, nfft=1024, window='hamming', fs=1000,
                                        remove_static=True)
    batchComputer.compute_save_all_stft(nperseg=1024, noverlap=256, nfft=2048, window='hamming', fs=1000,
                                        remove_static=False)
    batchComputer.compute_save_all_stft(nperseg=1024, noverlap=256, nfft=2048, window='hamming', fs=1000,
                                        remove_static=True)
