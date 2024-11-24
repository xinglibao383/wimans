import numpy as np
import warnings
from sklearn.decomposition import FastICA
from scipy.stats import pearsonr


def blind_source_separation(csi_data, num_users, random_state=0):
    warnings.filterwarnings("ignore", category=UserWarning, module='sklearn.decomposition._fastica')

    time, transmitters, receivers, subcarriers = csi_data.shape
    separated_csi_data = np.zeros((num_users, time, transmitters, receivers, subcarriers))
    ica = FastICA(n_components=num_users, random_state=random_state)

    for tx in range(0, transmitters):
        for rx in range(0, receivers):
            data = csi_data[:, tx, rx, :]
            result = np.zeros((subcarriers, time, num_users))
            for subcarrier in range(subcarriers):
                result[subcarrier, :, :] = ica.fit_transform(data[:, subcarrier].reshape(-1, 1))
            separated_csi_data[:, :, tx, rx, :] = np.transpose(result, (2, 1, 0))

    return separated_csi_data


def invert(csi_data):
    inverted_csi_data = np.zeros_like(csi_data)
    users, time, transmitters, receivers, subcarriers = csi_data.shape

    for user in range(0, users):
        inverted_data = np.zeros((time, transmitters, receivers, subcarriers))
        for tx in range(0, transmitters):
            for rx in range(0, receivers):
                data = csi_data[user, :, tx, rx, :]
                avg = np.mean(data, axis=1).reshape(-1, 1)
                inverted_data[:, tx, rx, :] = np.abs(data - 2 * avg)
        inverted_csi_data[user, :, :, :, :] = inverted_data

    return inverted_csi_data


def calculate_correlation_coefficient_and_record(separated_csi_data, inverted_csi_data):
    recorded_csi_data = np.zeros_like(separated_csi_data)
    num_users, time, transmitters, receivers, subcarriers = separated_csi_data.shape

    for user in range(0, num_users):
        for tx in range(0, transmitters):
            for rx in range(0, receivers):
                recorded_csi_data[user, :, tx, rx, 0] = separated_csi_data[user, :, tx, rx, 0]
                for subcarrier in range(1, subcarriers):
                    prev_data = recorded_csi_data[user, :, tx, rx, subcarrier - 1]
                    max_corr, best_data = -1, None
                    for candidate in range(0, num_users):
                        corr, _ = pearsonr(prev_data, separated_csi_data[candidate, :, tx, rx, subcarrier])
                        if corr > max_corr:
                            max_corr, best_data = corr, separated_csi_data[user, :, tx, rx, subcarrier]
                    for candidate in range(0, num_users):
                        corr, _ = pearsonr(prev_data, inverted_csi_data[candidate, :, tx, rx, subcarrier])
                        if corr > max_corr:
                            max_corr, best_data = corr, inverted_csi_data[user, :, tx, rx, subcarrier]
                    recorded_csi_data[user, :, tx, rx, subcarrier] = best_data

    return recorded_csi_data


def ccr_ica(csi_data, num_users):
    separated_csi_data = blind_source_separation(csi_data, num_users)
    inverted_csi_data = invert(separated_csi_data)
    return calculate_correlation_coefficient_and_record(separated_csi_data, inverted_csi_data)


if __name__ == "__main__":
    time, transmitters, receivers, subcarriers, num_users = 3000, 3, 3, 30, 4
    csi_data = np.random.rand(time, transmitters, receivers, subcarriers)
    recorded_csi_data = ccr_ica(csi_data, num_users)
    print(recorded_csi_data.shape)
