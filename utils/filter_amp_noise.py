"""
Noise Filtering: To mitigate noise in CSI data caused by fluctuating transmission power and rates, 
we apply the Hampel filter [3] to remove outliers from the CSI amplitude, 
followed by wavelet denoising to separate the signal from in-band noise and reconstruct clean CSI signals. 
Additionally, given that human activity affects CSI in the 20-60 Hz range [24], [25], 
we use a Butterworth filter to eliminate high-frequency noise while preserving low-frequency signals associated with human movement.
"""
import os
import numpy as np
from scipy.signal import butter, filtfilt
import pywt
from statsmodels.robust.scale import mad


# 使用Hampel滤波器去除CSI振幅中的离群值
def hampel_filter(data, window_size, n_sigmas=3):
    filtered_data = np.copy(data)
    half_window = window_size // 2

    for i in range(half_window, len(data) - half_window):
        window = data[i - half_window:i + half_window + 1]
        median = np.median(window)
        mad_value = mad(window)
        threshold = n_sigmas * mad_value

        if abs(data[i] - median) > threshold:
            filtered_data[i] = median

    return filtered_data


# 通过小波去噪将信号与带内噪声分离，重建干净的CSI信号
def wavelet_denoise(data, wavelet='db4', level=2):
    coeffs = pywt.wavedec(data, wavelet, mode='symmetric')
    threshold = np.sqrt(2 * np.log(len(data))) * mad(coeffs[-1])
    denoised_coeffs = [pywt.threshold(c, value=threshold, mode='soft') for c in coeffs]
    return pywt.waverec(denoised_coeffs, wavelet, mode='symmetric')[:len(data)]


# 使用巴特沃斯滤波器去除高频噪声
def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)


def filter_noise(filepath, save_path):
    data = np.load(filepath)

    # 对子载波进行Hampel滤波
    for t in range(data.shape[0]):
        for tx in range(data.shape[1]):
            for rx in range(data.shape[2]):
                data[t, tx, rx, :] = hampel_filter(data[t, tx, rx, :], window_size=11)

    # 对子载波进行小波去噪
    for t in range(data.shape[0]):
        for tx in range(data.shape[1]):
            for rx in range(data.shape[2]):
                data[t, tx, rx, :] = wavelet_denoise(data[t, tx, rx, :])

    for t in range(data.shape[0]):
        for tx in range(data.shape[1]):
            for rx in range(data.shape[2]):
                data[t, tx, rx, :] = butter_bandpass_filter(data[t, tx, rx, :], lowcut=20, highcut=60, fs=1000)

    np.save(save_path, data)
    return f'Success: {save_path}'


def filter_all_amp_noise(parent_path, save_parent_path):
    if not os.path.exists(save_parent_path):
        os.makedirs(save_parent_path)

    for root, _, files in os.walk(parent_path):
        for file in files:
            if file.endswith('.npy'):
                info = filter_noise(os.path.join(root, file), os.path.join(save_parent_path, file))
                print(info)


if __name__ == "__main__":
    filter_all_amp_noise(r"E:\WorkSpace\WiMANS\dataset\wifi_csi\amp",
                         r"E:\WorkSpace\WiMANS\dataset\wifi_csi\amp_without_noise")
