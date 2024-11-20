import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def pca_reduce_data(csi_data, n_components=12):
    time, transmitter, receiver, subcarrier = csi_data.shape
    reduced_data = np.zeros((time, transmitter, receiver, n_components), dtype=np.float32)

    for i in range(transmitter):
        for j in range(receiver):
            data = csi_data[:, i, j, :]

            scaler = StandardScaler()
            data_scaled = scaler.fit_transform(data)

            pca = PCA(n_components=n_components)
            data_reduced = pca.fit_transform(data_scaled)

            reduced_data[:, i, j, :] = data_reduced

    return reduced_data


if __name__ == "__main__":
    csi_data = np.load(r'E:\WorkSpace\WiMANS\dataset\wifi_csi\amp_without_noise\act_1_1.npy')
    reduced_data = pca_reduce_data(csi_data, n_components=12)
    print(reduced_data.shape)
