import os
import json
import datetime
import torch
from wimans_v3 import WiMANS, get_dataloaders
from tools.logger import Logger
from models.model_v1 import MyModel as MyModelV1
from models.model_v2 import MyModel as MyModelV2
import trainV4_dual

def prepare():
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    output_save_path = os.path.join('./', 'outputs', timestamp)
    logger = Logger(save_path=output_save_path)

    devices = [torch.device('cuda:0'), torch.device('cuda:1'), torch.device('cuda:2'), torch.device('cuda:3')]

    return output_save_path, logger, devices


def train(nperseg, noverlap, nfft, window, remove_static, remove_noise, 
          hidden_dim, nhead, encoder_layers, dropout1, dropout2, dropout3,
          learning_rate, weight_decay, use_scheduler, task, 
          feature_extractor1_name, feature_extractor2_name):
    output_save_path, logger, devices = prepare()
    params = {
        'nperseg': nperseg,
        'noverlap': noverlap,
        'nfft': nfft,
        'window': window,
        'remove_static': remove_static,
        'remove_noise': remove_noise,
        'hidden_dim': hidden_dim,
        'nhead': nhead,
        'encoder_layers': encoder_layers,
        'dropout1': dropout1,
        'dropout2': dropout2,
        'dropout3': dropout3,
        'learning_rate': learning_rate,
        'weight_decay': weight_decay,
        'use_scheduler': use_scheduler,
        'task': task,
        'feature_extractor1_name': feature_extractor1_name,
        'feature_extractor2_name': feature_extractor2_name
    }
    logger.record([f'Training parameters: {json.dumps(params, indent=4)}'])
    
    dataset = WiMANS(root_path='/data/XLBWorkSpace/wimans', 
                     nperseg=nperseg, noverlap=noverlap, nfft=nfft, window=window, remove_static=remove_static, remove_noise=remove_noise)
    train_loader, val_loader = trainV4_dual.get_dataloaders_without_test(dataset, batch_size=64)

    # net = MyModelV1(hidden_dim=hidden_dim, nhead=nhead, encoder_layers=encoder_layers, dropout1=dropout1, dropout2=dropout2, dropout3=dropout3)
    net = MyModelV2(hidden_dim=hidden_dim, nhead=nhead, encoder_layers=encoder_layers, dropout1=dropout1, dropout2=dropout2, dropout3=dropout3, 
                    feature_extractor1_name=feature_extractor1_name, feature_extractor2_name=feature_extractor2_name)

    trainV4_dual.train(net, train_loader, val_loader, learning_rate, weight_decay, 500, 50, 
                       devices, output_save_path, logger, use_scheduler=use_scheduler, task=task)
    

if __name__ == "__main__":
      train(nperseg=512, noverlap=256, nfft=1024, window='hamming', remove_static=True, remove_noise=True, 
            hidden_dim=1024, nhead=8, encoder_layers=8, dropout1=0.3, dropout2=0.3, dropout3=0.1,
            learning_rate=0.0001, weight_decay=1e-4, use_scheduler=False, task='3', 
            feature_extractor1_name='temporal_fusion_transformer', feature_extractor2_name='swin_transformer')