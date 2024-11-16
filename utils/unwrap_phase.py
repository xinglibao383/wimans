import os
import numpy as np


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
    unwrap_all_phase(r"E:\WorkSpace\WiMANS\dataset\wifi_csi\phase", 
                     r"E:\WorkSpace\WiMANS\dataset\wifi_csi\phase_unwrapped")