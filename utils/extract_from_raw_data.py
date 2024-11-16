import os
import numpy as np
import scipy.io as scio

# 从CSI提取幅度
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

# 从CSI提取相位
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


# 相位展开
def unwrap_phase(filepath, save_path):
    phase = np.load(filepath)
    unwrapped_phase = np.unwrap(phase, axis=-1)
    np.save(save_path, unwrapped_phase)
    return f'Success: {save_path}'


def unwrap_all_phase(parent_path, save_parent_path):
    if not os.path.exists(save_parent_path):
        os.makedirs(save_parent_path)

    for root, _, files in os.walk(parent_path):
        for file in files:
            if file.endswith('.npy'):
                info = unwrap_phase(os.path.join(root, file), os.path.join(save_parent_path, file))
                print(info)




if __name__ == "__main__":
    extract_csi_phase(var_dir_mat=r'E:\WorkSpace\WiMANS\dataset\wifi_csi\mat',
                      var_dir_phase=r'E:\WorkSpace\WiMANS\dataset\wifi_csi\phase')
