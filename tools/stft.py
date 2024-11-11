import os
import numpy as np
from scipy.signal import stft


def compute_stft(csi_file_path, fs=1000, nfft=2048, nperseg=512, noverlap=256, window='hamming'):
    csi_data = np.load(csi_file_path)
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


def compute_save_all_stft(parent_path, save_parent_path,
                          fs=1000, nfft=2048, nperseg=512, noverlap=256, window='hamming'):
    if not os.path.exists(save_parent_path):
        os.makedirs(save_parent_path)

    for filename in os.listdir(parent_path):
        if filename.endswith('.npy'):
            Zxx_270 = compute_stft(os.path.join(parent_path, filename),
                                   fs=fs, nfft=nfft, nperseg=nperseg, noverlap=noverlap, window=window)
            save_path = os.path.join(save_parent_path, filename)
            np.save(save_path, Zxx_270)
            print(f"Saved STFT result for {filename} to {save_path}")


if __name__ == "__main__":
    compute_save_all_stft(r'E:\WorkSpace\WiMANS\dataset\wifi_csi\amp',
                          r'E:\WorkSpace\WiMANS\dataset\wifi_csi\stft')
