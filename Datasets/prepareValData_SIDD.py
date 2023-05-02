#!/usr/bin/env python
# -*- coding:utf-8 -*-

# load .mat val image data into h5

import argparse
from scipy.io import loadmat
import os
import h5py as h5
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True, 
                        help='data path for SIDD val dataset. Please put noisy and clean files under same folder')
    parser.add_argument('--output_path', type=str, default='./', help='dataset saving path')

    args = parser.parse_args()

    # ***************************************
    # step 1: load val noisy and val gt files
    # ***************************************
    print(f'Loading val files...')
    data_path = args.data_path
    val_noisy_file = loadmat(os.path.join(data_path, 'ValidationNoisyBlocksSrgb.mat'))
    val_noisy_imgs = val_noisy_file['ValidationNoisyBlocksSrgb']
    val_gt_file = loadmat(os.path.join(data_path, 'ValidationGtBlocksSrgb.mat'))
    val_gt_imgs = val_gt_file['ValidationGtBlocksSrgb']

    print(f'Val data file loads complete.')

    # ******************************************
    # Step 2: slice original images into patches
    # ******************************************
    output_path = args.output_path

    h5_file = os.path.join(output_path, 'valDataset.hdf5')
    if os.path.exists(h5_file):
        os.remove(h5_file)
    
    print(f'Saving validation image samples into {h5_file}.')
    # SIDD Validation Data and Ground Truth as single .mat arrays of dimensoins [#images, #blocks, height, width, #channels]
    num_imgs, num_blocks, height, width, num_channels = val_noisy_imgs.shape
    print(f'Number of total noisy & gt image blocks: {num_imgs} * {num_blocks} = {num_imgs * num_blocks}')
    sample_idx = 0
    with h5.File(h5_file, 'w') as output:
        for img_idx in range(num_imgs):
            if img_idx % 20 == 0:
                print(f'####Processing the No.{img_idx} image...')
            for block_idx in range(num_blocks):
                noisy_blk = val_noisy_imgs[img_idx][block_idx]
                gt_blk = val_gt_imgs[img_idx][block_idx]
                patch_sample = np.concatenate((noisy_blk, gt_blk), axis=2)
                output.create_dataset(name=str(sample_idx), shape=patch_sample.shape, 
                                          dtype=patch_sample.dtype, data=patch_sample)
                
                sample_idx += 1

    
    print(f'Load total {sample_idx} pairs of noisy and gt image blocks')
    print('Done!')



if __name__ == '__main__':
    main()