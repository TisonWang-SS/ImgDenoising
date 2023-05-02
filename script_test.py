import torch
from torch import nn
from torch.utils.data import DataLoader
import os
import numpy as np

from DenoisingDatasets import SIDDTrainDataset, SIDDValDataset
from data_augmentation import get_train_transforms, get_valid_transforms

from tqdm import tqdm

params = {
    'device': torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'),
    'seed': 415,
    'lr': 1e-3,
    'batch_size': 16,
    'num_workers': 0,
    'epochs': 50,
    'weight_decay': 1e-5,

    'train_data_file': r'./Datasets/trainDataset.hdf5',
    'val_data_file': r'./Datasets/valDataset.hdf5',

    'ndf': 64,
    'alpha': 0.5,
    'kernel_size': 5,
    "lambda_gp": 10,
    "tau_R": 1000,
    "tau_G": 10,
    "num_critic": 3
}

train_dataset = SIDDTrainDataset(params['train_data_file'], transform=get_train_transforms(128))
val_dataset = SIDDValDataset(params['val_data_file'], transform=get_valid_transforms(128))

# train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True, 
#                               num_workers=params['num_workers'], pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=params['batch_size'], shuffle=False, 
                            num_workers=params['num_workers'], pin_memory=True)

# data_stream = tqdm(train_loader)

for i, (x, y) in enumerate(val_loader):
    x1 = x
    y1 = y
    break