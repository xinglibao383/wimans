import os
import numpy as np
import scipy.io as scio


def mat_to_amp(data_mat):
    var_length = data_mat["trace"].shape[0]
    data_csi_amp = [abs(data_mat["trace"][var_t][0][0][0][-1]) for var_t in range(var_length)]
    data_csi_amp = np.array(data_csi_amp, dtype=np.float32)
    return data_csi_amp


def extract_csi_amp(var_dir_mat, var_dir_amp):
    var_path_mat = os.listdir(var_dir_mat)
    for var_c, var_path in enumerate(var_path_mat):
        data_mat = scio.loadmat(os.path.join(var_dir_mat, var_path))
        data_csi_amp = mat_to_amp(data_mat)
        print(var_c, data_csi_amp.shape)
        var_path_save = os.path.join(var_dir_amp, var_path.replace(".mat", ".npy"))
        with open(var_path_save, "wb") as var_file:
            np.save(var_file, data_csi_amp)


def mat_to_phase(data_mat):
    var_length = data_mat["trace"].shape[0]
    data_csi_phase = [np.angle(data_mat["trace"][var_t][0][0][0][-1]) for var_t in range(var_length)]
    data_csi_phase = np.array(data_csi_phase, dtype=np.float32)
    return data_csi_phase


def extract_csi_phase(var_dir_mat, var_dir_phase):
    var_path_mat = os.listdir(var_dir_mat)
    for var_c, var_path in enumerate(var_path_mat):
        data_mat = scio.loadmat(os.path.join(var_dir_mat, var_path))
        data_csi_phase = mat_to_phase(data_mat)
        print(var_c, data_csi_phase.shape)
        var_path_save = os.path.join(var_dir_phase, var_path.replace(".mat", ".npy"))
        with open(var_path_save, "wb") as var_file:
            np.save(var_file, data_csi_phase)


if __name__ == "__main__":
    extract_csi_phase(var_dir_mat=r'E:\WorkSpace\WiMANS\dataset\wifi_csi\mat',
                      var_dir_phase=r'E:\WorkSpace\WiMANS\dataset\wifi_csi\phase')
