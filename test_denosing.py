import torch
import argparse
from os.path import join
from network import UNetR
import numpy as np
import torch.nn as nn
from scipy.io import loadmat
from skimage import img_as_float32, img_as_ubyte
# from skimage.measure import compare_psnr, compare_ssim
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser()
args = parser.parse_args()

noisy_path = './Datasets/SIDD_Val/ValidationNoisyBlocksSrgb.mat'
gt_path = './Datasets/SIDD_Val/ValidationGtBlocksSrgb.mat'

noisy_imgs = loadmat(noisy_path)['ValidationNoisyBlocksSrgb'] # uint8 format
gt_imgs = loadmat(gt_path)['ValidationGtBlocksSrgb']

denoised_imgs = np.zeros_like(gt_imgs)
num_img, num_block, _, _, _ = noisy_imgs.shape
total_blocks = num_img * num_block

# load the network
netD = UNetR(in_channels=3, out_channels=3, depth=5, wf=32).cuda()
netD.load_state_dict(torch.load('./checkpoints/model_50epochs_weights.pth', map_location='cpu')['R'])
netD.eval()

psnr = ssim = 0
counter = 0
for ii in range(num_img):
    for jj in range(num_block):
        noisy_im_iter = img_as_float32(noisy_imgs[ii, jj,].transpose((2,0,1))[np.newaxis,])
        gt_im_iter = gt_imgs[ii, jj,]
        with torch.no_grad():
            inputs = torch.from_numpy(noisy_im_iter).cuda()
            outputs = inputs - netD(inputs)
            outputs.clamp_(0.0, 1.0)
            outputs = outputs.cpu().numpy().squeeze().transpose((1,2,0))
        
        plt.subplot(1,3,1)
        plt.imshow(noisy_imgs[ii, jj,])
        plt.title('Noisy Image')
        plt.axis('off')
        plt.subplot(1,3,2)
        plt.imshow(gt_im_iter)
        plt.title('Gt Image')
        plt.axis('off')
        plt.subplot(1,3,3)
        plt.imshow(outputs)
        plt.title('Denoised Image')
        plt.axis('off')
        plt.show()
        break
        denoised_im_iter = img_as_ubyte(outputs)
        denoised_imgs[ii, jj,] = denoised_im_iter
        psnr_iter = compare_psnr(denoised_im_iter, gt_im_iter)
        psnr += psnr_iter
        ssim_iter = compare_ssim(denoised_im_iter, gt_im_iter, data_range=255, gaussian_weights=True,
                                                     use_sample_covariance=False, multichannel=True)
        ssim += ssim_iter

        counter += 1
        if counter % 50 == 0:
            print('{:04d}/{:04d}, psnr={:.2f}, ssim={:.4f}'.format(counter, total_blocks,
                                                                              psnr_iter, ssim_iter))

# psnr /= total_blocks
# ssim /= total_blocks
# print('Finish: PSNR = {:.2f}, SSIM={:4f}'.format(psnr, ssim))