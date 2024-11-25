import os
import numpy as np
import pandas as pd
from ccr_ica import ccr_ica


def get_labels_from_csv(csv, row_id):
    location_map = {
        'a': 1,
        'b': 2,
        'c': 3,
        'd': 4,
        'e': 5,
    }
    activity_map = {
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
    row = csv[csv['label'] == row_id]
    return row['number_of_users'].values[0]
    

def get_filepaths_and_labels(root_path):
    filenames = [f for f in os.listdir(os.path.join(root_path, 'wifi_csi', 'amp')) if f.endswith('.npy')]
    csv = pd.read_csv(os.path.join(root_path, 'annotation.csv'))
    filepaths = [os.path.join(root_path, 'wifi_csi', 'amp', f) for f in filenames]
    labels = [get_labels_from_csv(csv, f[:-4]) for f in filenames]

    return filenames, filepaths, labels


if __name__ == "__main__":
    root_path = '/data/XLBWorkSpace/wimans'
    save_path = '/data/temp/wimans'
    filenames, filepaths, labels = get_filepaths_and_labels(root_path)
    for i in range(0, len(filenames)):
        if labels[i] == 0:
            np.save(os.path.join(save_path, 'wifi_csi', 'amp_ccr_ica', filenames[i]), np.empty((0, 3000, 3, 3, 30)))
        else:
            data = ccr_ica(np.load(filepaths[i]), labels[i])
            np.save(os.path.join(save_path, 'wifi_csi', 'amp_ccr_ica', filenames[i]), data)
        print(os.path.join(save_path, 'wifi_csi', 'amp_ccr_ica', filenames[i]))
