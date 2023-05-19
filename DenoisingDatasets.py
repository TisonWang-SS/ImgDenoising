#!/usr/bin/env python
# -*- coding:utf-8 -*-

# Denoising Dataset

from torch.utils.data import Dataset
import h5py as h5
import torch
from skimage import img_as_float32 as img_as_float
from data_augmentation import random_augmentation


class SIDDTrainDataset(Dataset):
    '''
    Train Dataset of SIDD
    '''
    def __init__(self, h5_path, patch_size=128, transform=None):
        '''
        init a SIDDTrainDataset Object with default patch_size=128
        '''
        super().__init__()
        self.h5_path = h5_path
        self.patch_size = patch_size
        self.transform = transform
        self.img_size = 512
        self.chs = 3
        self.num_patches_per_img = (self.img_size // patch_size)**2
        with h5.File(h5_path, 'r') as h5_file:
            self.keys = list(h5_file.keys())
            self.num_images = len(h5_file.keys())
            self.length = self.num_images * self.num_patches_per_img
    
    def __len__(self):
        '''
        get length of dataset
        '''
        return self.length

    def __getitem__(self, index):
        '''
        get train image patch pairs on given index
        image patch is cropped on the original SIDD images
        '''
        img_idx = index // self.num_patches_per_img
        patch_idx = index % self.num_patches_per_img
        patch_idx_h = patch_idx // (self.img_size // self.patch_size )
        patch_idx_w = patch_idx % (self.img_size // self.patch_size )
        with h5.File(self.h5_path, 'r') as h5_file:
            img_noisy = h5_file[self.keys[img_idx]][:,:,:self.chs]
            img_gt = h5_file[self.keys[img_idx]][:,:,self.chs:]

            start_h = patch_idx_h * self.patch_size
            start_w = patch_idx_w * self.patch_size
            patch_noisy = img_noisy[start_h : start_h + self.patch_size, 
                                    start_w : start_w + self.patch_size, :]
            patch_gt = img_gt[start_h : start_h + self.patch_size, 
                                    start_w : start_w + self.patch_size, :]
        
        patch_noisy = img_as_float(patch_noisy)
        patch_gt = img_as_float(patch_gt)

        patch_gt, patch_noisy = random_augmentation(patch_gt, patch_noisy)
        
        if self.transform is not None:
            patch_noisy = self.transform(image=patch_noisy)['image']
            patch_gt = self.transform(image=patch_gt)['image']
        else:
            patch_noisy = torch.from_numpy(patch_noisy.transpose((2, 0, 1)))
            patch_gt = torch.from_numpy(patch_gt.transpose((2, 0, 1)))
        
        return patch_noisy, patch_gt


class SIDDValDataset(Dataset):
    '''
    Valid Dataset of SIDD
    '''
    def __init__(self, h5_path, transform=None):
        '''
        init a SIDDValDataset Object with default patch_size=128
        '''
        super().__init__()
        self.h5_path = h5_path
        self.transform = transform
        self.chs = 3
        with h5.File(h5_path, 'r') as h5_file:
            self.keys = list(h5_file.keys())
            self.num_images = len(h5_file.keys())
            self.length = self.num_images
    
    def __len__(self):
        '''
        get length of dataset
        '''
        return self.length
        
    def __getitem__(self, index):
        '''
        get valid image patch pairs on given index
        image patch is cropped on the original SIDD images
        '''
        with h5.File(self.h5_path, 'r') as h5_file:
            img_noisy = h5_file[self.keys[index]][:,:,:self.chs]
            img_gt = h5_file[self.keys[index]][:,:,self.chs:]
        
        img_noisy = img_as_float(img_noisy)
        img_gt = img_as_float(img_gt)

        if self.transform is not None:
            img_noisy = self.transform(image=img_noisy)['image']
            img_gt = self.transform(image=img_gt)['image']
        else:
            img_noisy = torch.from_numpy(img_noisy.transpose((2, 0, 1)))
            img_gt = torch.from_numpy(img_gt.transpose((2, 0, 1)))
        
        return img_noisy, img_gt
