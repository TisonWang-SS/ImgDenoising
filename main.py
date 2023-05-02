#!/usr/bin/env python
# -*- coding:utf-8 -*-

import torch
from torch import nn
from torch.utils.data import DataLoader
import os
import numpy as np

from DenoisingDatasets import SIDDTrainDataset, SIDDValDataset
from data_augmentation import get_train_transforms, get_valid_transforms

from network import UNetR, UNetG, Discriminator
from loss import get_gausskernel

from train import train, validate

params = {
    'device': torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'),
    'seed': 415,
    'lr': 1e-4,
    'batch_size': 16,
    'val_batch_size': 4,
    'num_workers': 4,
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
    "num_critic": 3,

    'checkpoints_path': r'./checkpoints/'
}

def main():
    seed = params['seed']

    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    train_dataset = SIDDTrainDataset(params['train_data_file'], transform=None)
    val_dataset = SIDDValDataset(params['val_data_file'], transform=None)

    train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True, 
                              num_workers=params['num_workers'], pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=params['val_batch_size'], shuffle=False, 
                            num_workers=params['num_workers'], pin_memory=True)
    
    netR = UNetR(in_channels=3, out_channels=3, depth=5, wf=32).to(params['device'])
    netG = UNetG(in_channels=3, out_channels=3, depth=5, wf=32).to(params['device'])
    netD = Discriminator(6, 64).to(params['device'])
    net = {
        'R': netR,
        'G': netG,
        'D': netD
    }

    optimizerR = torch.optim.AdamW(netR.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
    optimizerG = torch.optim.AdamW(netG.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
    optimizerD = torch.optim.AdamW(netD.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
    optimizer = {
        'R':optimizerR, 
        'G':optimizerG, 
        'D':optimizerD
    }

    kernel_size = params['kernel_size']

    global kernel
    kernel = get_gausskernel(kernel_size)

    if not os.path.exists(params['checkpoints_path']):
        os.mkdir(params['checkpoints_path'])

    print(f'Training on {params["device"]}.')

    for epoch in range(1, params['epochs'] + 1):
        train(net, train_loader, optimizer, params, epoch, kernel)
        validate(net, val_loader, params, epoch)
        torch.save(
            {
                'R': net['R'].state_dict(),
                'G': net['G'].state_dict(),
                'D': net['D'].state_dict(),
            },
            os.path.join(params['checkpoints_path'], f'model_{epoch}epochs_weights.pth')
            # f"./checkpoints/model_{epoch}epochs_weights.pth"
        )
    
    print('-' * 60)
    print('Done!')

if __name__ == '__main__':
    main()