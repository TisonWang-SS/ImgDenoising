import torch
from network import UNetR
import numpy as np
from scipy.io import loadmat
from skimage import img_as_float32
from matplotlib import pyplot as plt
import cv2

# load validation images
noisy_path = './Datasets/SIDD_Val/ValidationNoisyBlocksSrgb.mat'
gt_path = './Datasets/SIDD_Val/ValidationGtBlocksSrgb.mat'

noisy_imgs = loadmat(noisy_path)['ValidationNoisyBlocksSrgb']
gt_imgs = loadmat(gt_path)['ValidationGtBlocksSrgb']

num_img, num_block, _, _, _ = noisy_imgs.shape

# load the network
netD = UNetR(in_channels=3, out_channels=3, depth=5, wf=32).cuda()
netD.load_state_dict(torch.load('./checkpoints/2023_05_16_23_45_00_model_50epochs_weights.pth', map_location='cpu')['R'])
netD.eval()

# results display
for ii in range(num_img):
    for jj in range(num_block):
        noisy_im_iter = img_as_float32(noisy_imgs[ii, jj,].transpose((2,0,1))[np.newaxis,])
        gt_im_iter = gt_imgs[ii, jj,]
        with torch.no_grad():
            inputs = torch.from_numpy(noisy_im_iter).cuda()
            outputs = inputs - netD(inputs)
            outputs.clamp_(0.0, 1.0)
            outputs = outputs.cpu().numpy().squeeze().transpose((1,2,0))
        
        # create a detail addback image with a mix of noisy image and denoised image
        output_l = cv2.cvtColor(outputs, cv2.COLOR_RGB2Luv)
        noisy_im_iter_l = cv2.cvtColor(noisy_imgs[ii, jj,], cv2.COLOR_RGB2Luv)
        detail_addback_l = output_l
        detail_addback_l[:,:,0] = output_l[:,:,0] * 0.95 + noisy_im_iter_l[:,:,0] * 0.05
        detail_addback = cv2.cvtColor(detail_addback_l, cv2.COLOR_Luv2RGB)
        
        # plot
        plt.subplot(1,4,1)
        plt.imshow(noisy_imgs[ii, jj,])
        plt.title('Noisy Image')
        plt.axis('off')
        plt.subplot(1,4,2)
        plt.imshow(gt_im_iter)
        plt.title('Gt Image')
        plt.axis('off')
        plt.subplot(1,4,3)
        plt.imshow(outputs)
        plt.title('Denoised Image')
        plt.axis('off')
        plt.subplot(1,4,4)
        plt.imshow(detail_addback)
        plt.title('Detail Addback')
        plt.axis('off')
        plt.show()
