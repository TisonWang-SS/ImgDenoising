#!/usr/bin/env python
# -*- coding:utf-8 -*-


import torch
import torch.nn.functional as F
import cv2
import numpy as np
import math
from skimage import img_as_ubyte
from network import sample_generator

# kernel = get_gausskernel(5, chn=3)


def loss_UNetR(net, x, y, x_hat, params):
    alpha = params['alpha']
    tau_R = params['tau_R']

    mae_loss = F.l1_loss(x_hat, x, reduction='mean')
    fake_x_data = torch.cat([x_hat, y], 1)
    fake_x_loss = net['D'](fake_x_data).mean()
    loss_x = -(1 - alpha) * fake_x_loss
    loss_e = tau_R * mae_loss
    loss = loss_x + loss_e

    return loss

def loss_UNetG(net, x, y, y_hat, params, kernel):
    alpha = params['alpha']
    tau_G = params['tau_G']
    # kernel_size = params['kernel_size']

    # kernel = get_gausskernel(kernel_size)

    loss_mean = tau_G * mean_match(x, y, y_hat, kernel.to(x.device))
    fake_y_data = torch.cat([x, y_hat], 1)
    fake_y_loss = net['D'](fake_y_data).mean()
    loss_y = -alpha * fake_y_loss
    loss = loss_y + loss_mean

    return loss

def loss_Discriminator(net, x, y, real_loss, params):
    alpha = params['alpha']
    real_data = torch.cat([x, y], 1)
    lambda_gp = params['lambda_gp']

    # generator fake data
    with torch.autograd.no_grad():
        fake_y = sample_generator(net['G'], x)
        fake_y_data = torch.cat([x, fake_y], 1)
    fake_y_loss = net['D'](fake_y_data.data).mean()
    grad_y_loss = gradient_penalty(real_data, fake_y_data, net['D'], lambda_gp)
    loss_y = alpha * (fake_y_loss - real_loss)
    loss_yg = alpha * grad_y_loss
    # Denoiser fake data
    with torch.autograd.no_grad():
        fake_x = y - net['R'](y)
        fake_x_data = torch.cat([fake_x, y], 1)
    fake_x_loss = net['D'](fake_x_data.data).mean()
    grad_x_loss = gradient_penalty(real_data, fake_x_data, net['D'], lambda_gp)
    loss_x = (1-alpha) * (fake_x_loss - real_loss)
    loss_xg = (1-alpha) * grad_x_loss

    loss = loss_x + loss_xg + loss_y + loss_yg

    return loss

def get_gausskernel(p, chn=3):
    '''
    Build a 2-dimensional Gaussian filter with size p
    '''
    x = cv2.getGaussianKernel(p, sigma=-1)   # p x 1
    y = np.matmul(x, x.T)[np.newaxis, np.newaxis,]  # 1x 1 x p x p
    out = np.tile(y, (chn, 1, 1, 1)) # chn x 1 x p x p

    return torch.from_numpy(out).type(torch.float32)

def gaussblur(x, kernel, p=5, chn=3):
    x_pad = F.pad(x, pad=[int((p-1)/2),]*4, mode='reflect')
    y = F.conv2d(x_pad, kernel, padding=0, stride=1, groups=chn)

    return y

def mean_match(x, y, fake_y, kernel, chn=3):
    p = kernel.shape[2]
    # estimate the real distribution
    err_real = y - x
    mu_real = gaussblur(err_real, kernel, p, chn)
    err_fake = fake_y - x
    mu_fake = gaussblur(err_fake, kernel, p, chn)
    loss = F.l1_loss(mu_real, mu_fake, reduction='mean')

    return loss

def gradient_penalty(real_data, generated_data, netP, lambda_gp):
    batch_size = real_data.size()[0]

    # Calculate interpolation
    alpha = torch.rand(batch_size, 1, 1, 1)
    alpha = alpha.expand_as(real_data)
    alpha = alpha.to(real_data.device)
    interpolated = alpha * real_data.data + (1 - alpha) * generated_data.data
    interpolated.requires_grad=True

    # Calculate probability of interpolated examples
    prob_interpolated = netP(interpolated)

    # Calculate gradients of probabilities with respect to examples
    grad_outputs = torch.ones(prob_interpolated.size(), device=real_data.device, dtype=torch.float32)
    gradients = torch.autograd.grad(outputs=prob_interpolated, inputs=interpolated, 
                                    grad_outputs=grad_outputs, create_graph=True, retain_graph=True)[0]

    # Gradients have shape (batch_size, num_channels, img_width, img_height),
    # so flatten to easily take norm per example in batch
    gradients = gradients.view(batch_size, -1)

    # Derivatives of the gradient close to 0 can cause problems because of
    # the square root, so manually calculate norm and add epsilon
    gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

    # Return gradient penalty
    return  lambda_gp * ((gradients_norm - 1) ** 2).mean()

def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

def calculate_ssim(img1, img2, border=0):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    h, w = img1.shape[:2]
    img1 = img1[border:h-border, border:w-border]
    img2 = img2[border:h-border, border:w-border]

    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1[:,:,i], img2[:,:,i]))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')

def calculate_psnr(im1, im2, border=0):
    if not im1.shape == im2.shape:
        raise ValueError('Input images must have the same dimensions.')
    h, w = im1.shape[:2]
    im1 = im1[border:h-border, border:w-border]
    im2 = im2[border:h-border, border:w-border]

    im1 = im1.astype(np.float64)
    im2 = im2.astype(np.float64)
    mse = np.mean((im1 - im2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))

def batch_PSNR(img, imclean, border=0):
    Img = img.data.cpu().numpy()
    Iclean = imclean.data.cpu().numpy()
    Img = img_as_ubyte(Img)
    Iclean = img_as_ubyte(Iclean)
    PSNR = 0
    for i in range(Img.shape[0]):
        PSNR += calculate_psnr(Iclean[i,:,].transpose((1,2,0)), Img[i,:,].transpose((1,2,0)), border)
    return (PSNR/Img.shape[0])

def batch_SSIM(img, imclean, border=0):
    Img = img.data.cpu().numpy()
    Iclean = imclean.data.cpu().numpy()
    Img = img_as_ubyte(Img)
    Iclean = img_as_ubyte(Iclean)
    SSIM = 0
    for i in range(Img.shape[0]):
        SSIM += calculate_ssim(Iclean[i,:,].transpose((1,2,0)), Img[i,:,].transpose((1,2,0)), border)
    return (SSIM/Img.shape[0])