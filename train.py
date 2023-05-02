#!/usr/bin/env python
# -*- coding:utf-8 -*-

from loss import loss_UNetR, loss_UNetG, loss_Discriminator, batch_PSNR, batch_SSIM
from network import sample_generator
import torch
from tqdm import tqdm
import math
import torch.nn.functional as F
from loss import *


def train_UNetR(net, x, y, optimizer, params):
    # for net_part in ['R', 'G', 'D']:
    #     net[net_part].to(params['device'])
    # x.to(params['device'])
    # y.to(params['device'])

    net['R'].zero_grad()
    x_hat = y - net['R'](y)

    # loss = loss_UNetR(net, x, y, x_hat, params)
    alpha = params['alpha']
    tau_R = params['tau_R']

    mae_loss = F.l1_loss(x_hat, x, reduction='mean')
    fake_x_data = torch.cat([x_hat, y], 1)
    fake_x_loss = net['D'](fake_x_data).mean()
    loss_x = -(1 - alpha) * fake_x_loss
    loss_e = tau_R * mae_loss
    loss = loss_x + loss_e

    loss.backward()
    optimizer.step()

    return loss, x_hat

def train_UNetG(net, x, y, optimizer, params, kernel):
    # for net_part in ['R', 'G', 'D']:
    #     net[net_part].to(params['device'])
    # x.to(params['device'])
    # y.to(params['device'])

    net['G'].zero_grad()
    y_hat = sample_generator(net['G'], x)

    # loss = loss_UNetG(net, x, y, y_hat, params, kernel)
    alpha = params['alpha']
    tau_G = params['tau_G']
    # kernel_size = params['kernel_size']

    # kernel = get_gausskernel(kernel_size)

    loss_mean = tau_G * mean_match(x, y, y_hat, kernel.to(x.device), 3)
    fake_y_data = torch.cat([x, y_hat], 1)
    fake_y_loss = net['D'](fake_y_data).mean()
    loss_y = -alpha * fake_y_loss
    loss = loss_y + loss_mean

    loss.backward()
    optimizer.step()

    return loss, y_hat

def train_Discriminator(net, x, y, optimizer, params):
    # for net_part in ['R', 'G', 'D']:
    #     net[net_part].to(params['device'])
    # x.to(params['device'])
    # y.to(params['device'])

    net['D'].zero_grad()

    real_data = torch.cat([x, y], 1)
    real_loss = net['D'](real_data).mean()

    # loss = loss_Discriminator(net, x, y, real_loss, params)
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

    loss.backward()
    optimizer.step()

    return loss, real_loss

def train(net, train_dl, optimizer, params, epoch, kernel):
    # for net_part in ['R', 'G', 'D']:
    #     net[net_part].to(params['device'])
    n_batches = len(train_dl)
    optimizer = optimizer

    data_stream = tqdm(train_dl)

    train_loss = {l : 0 for l in ['R', 'G', 'D']}

    for net_part in ['R', 'G', 'D']:
        net[net_part].train()

    for i, (imgs_noisy, imgs_gt) in enumerate(data_stream, start=1):
        imgs_noisy = imgs_noisy.to(params['device'], non_blocking=True)
        imgs_gt = imgs_gt.to(params['device'], non_blocking=True)

        loss_D, real_Loss = train_Discriminator(net, imgs_gt, imgs_noisy, optimizer['D'], params)

        train_loss['D'] += loss_D.item()

        if (i + 1) % params['num_critic'] == 0:
            # R
            loss_R, gt_hat = train_UNetR(net, imgs_gt, imgs_noisy, optimizer['R'], params)
            train_loss['R'] += loss_R.item()
            # G
            loss_G, noisy_hat = train_UNetG(net, imgs_gt, imgs_noisy, optimizer['G'], params, kernel)
            train_loss['G'] += loss_G.item()
    
    for op in ['R', 'G', 'D']:
        lr = adjust_learning_rate(optimizer[op], epoch, params, i, n_batches)
        
    train_loss_epoch = {l : train_loss[l] / (i + 1) for l in ['R', 'G', 'D']}
    UNetR_loss_epoch = train_loss_epoch['R']
    UNetG_loss_epoch = train_loss_epoch['G']
    Discriminator_loss_epoch = train_loss_epoch['D']
    print(f'#####Train##### Epoch: {epoch}, UNetR_Loss: {UNetR_loss_epoch:.4f}, UNetG_Loss: {UNetG_loss_epoch:.4f}, \
          Discriminator_Loss: {Discriminator_loss_epoch:.4f},')

def validate(net, val_dl, params, epoch):
    net['R'].eval()
    data_stream = tqdm(val_dl)

    mae_loss = []
    psnr_loss = []
    ssim_loss = []

    for i, (imgs_noisy, imgs_gt) in enumerate(data_stream, start=1):
        imgs_noisy = imgs_noisy.to(params['device'], non_blocking=True)
        imgs_gt = imgs_gt.to(params['device'], non_blocking=True)

        with torch.set_grad_enabled(False):
            gt_hat = imgs_noisy - net['R'](imgs_noisy)
        
        mae_loss.append(F.l1_loss(gt_hat, imgs_gt))
        gt_hat.clamp_(0.0, 1.0)
        psnr_loss.append(batch_PSNR(gt_hat, imgs_gt))
        ssim_loss.append(batch_SSIM(gt_hat, imgs_gt))
    
    mae_loss = sum(mae_loss) / len(mae_loss)
    psnr_loss = sum(psnr_loss) / len(psnr_loss)
    ssim_loss = sum(ssim_loss) / len(ssim_loss)

    print(f'#####Valid##### Epoch: {epoch}, mae_loss: {mae_loss:.4f}, psnr_loss: {psnr_loss:.4f}, \
          ssim_loss: {ssim_loss:.4f},')

def adjust_learning_rate(optimizer, epoch, params, batch=0, n_batch=None):
    """ adjust learning of a given optimizer and return the new learning rate """
    new_lr = calc_learning_rate(epoch, params['lr'], params['epochs'], batch, n_batch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr
    return new_lr

def calc_learning_rate(epoch, init_lr, n_epochs, batch=0, n_batch=None, lr_schedule_type='cosine'):
    """ learning rate schedule """
    if lr_schedule_type == 'cosine':
        t_total = n_epochs * n_batch
        t_cur = epoch * n_batch + batch
        lr = 0.5 * init_lr * (1 + math.cos(math.pi * t_cur / t_total))
    elif lr_schedule_type is None:
        lr = init_lr
    else:
        raise ValueError('do not support: %s' % lr_schedule_type)
    return lr