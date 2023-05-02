#!/usr/bin/env python
# -*- coding:utf-8 -*-

# transform png image data into h5

import argparse
from glob import glob
import os
import h5py as h5
import cv2
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True, help='data path for png image files')
    parser.add_argument('--output_path', type=str, default='./', help='dataset saving path')
    parser.add_argument('--patch_size', type=int, default=512, help='patch size of image samples')
    parser.add_argument('--overlap', type=int, default=128, help='overlap pixel length')

    args = parser.parse_args()

    # ****************************************
    # step 1: get all noisy & clean png images
    # ****************************************
    data_path = args.data_path
    noisy_image_paths = glob(os.path.join(data_path, '**/*NOISY*.PNG'), recursive=True)
    clean_image_paths = glob(os.path.join(data_path, '**/*GT*.PNG'), recursive=True)

    print(f'Number of noisy images: {len(noisy_image_paths)}, number of clean images: {len(clean_image_paths)}')

    if len(noisy_image_paths) != len(clean_image_paths):
        print('Error: number of noisy images and clean images are not equal.')
        exit(1)

    # sort images to make pairs
    sorted(noisy_image_paths)
    sorted(clean_image_paths)

    # ******************************************
    # Step 2: slice original images into patches
    # ******************************************
    patch_size = args.patch_size
    overlap = args.overlap
    output_path = args.output_path

    print(f'Crop original images into patches of {patch_size} with a overlap of {overlap}.')

    stride = patch_size - overlap

    h5_file = os.path.join(output_path, 'trainDataset.hdf5')
    if os.path.exists(h5_file):
        os.remove(h5_file)
    
    print(f'Saving image samples into {h5_file}.')
    sample_idx = 0
    with h5.File(h5_file, 'w') as output:
        for idx in range(len(noisy_image_paths)):
            if idx % 20 == 0:
                print(f'####Processing the No.{idx} image pair...')
            noisy_img = cv2.imread(noisy_image_paths[idx])
            noisy_img = noisy_img[::-1] # BGR -> RGB
            gt_img = cv2.imread(clean_image_paths[idx])
            gt_img = gt_img[::-1] # BGR -> RGB

            height, width, channel = noisy_img.shape

            start_h_pos = 0
            start_w_pos = 0

            while start_h_pos + patch_size < height:
                while start_w_pos + patch_size < width:
                    noisy_patch = noisy_img[start_h_pos : start_h_pos + patch_size, 
                                            start_w_pos : start_w_pos + patch_size, :]
                    gt_patch = gt_img[start_h_pos : start_h_pos + patch_size, 
                                      start_w_pos : start_w_pos + patch_size, :]
                    patch_sample = np.concatenate((noisy_patch, gt_patch), axis=2)
                    output.create_dataset(name=str(sample_idx), shape=patch_sample.shape, 
                                          dtype=patch_sample.dtype, data=patch_sample)
                    
                    sample_idx += 1
                    start_w_pos += stride

                start_h_pos += stride
    
    print(f'Crop total {sample_idx} pairs of noisy and gt image patches')
    print('Done!')



if __name__ == '__main__':
    main()