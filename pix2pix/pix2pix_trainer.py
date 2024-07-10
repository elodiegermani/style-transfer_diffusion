import time
from torch.autograd import Variable
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import numpy as np
import random
#from nilearn import plotting
import nibabel as nib 
from os.path import join as opj
import os 

def recreate_image(img_mat, affine, header, out_img = False):
    '''
    Recreate image based on an matrix, affine and header. 
    '''
    tmp_data = img_mat.detach().cpu().reshape(*img_mat.shape)
    
    if out_img:
        norm_img_data = tmp_data.copy().astype(float)
        norm_img_data = np.nan_to_num(norm_img_data)
        norm_img_data *= 1.0/np.abs(norm_img_data).max()
        img_data = norm_img_data
        
    else:
        img_data = tmp_data

    img = nib.Nifti1Image(img_data, affine=affine, header=header)

    return img

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm3d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

def Cor_CoeLoss(y_pred, y_target):
    data1 = np.reshape(y_pred.detach().cpu(), -1)
    data2 = np.reshape(y_target.detach().cpu(), -1)

    in_mask_indices = np.logical_not(
        np.logical_or(
            np.logical_or(np.isnan(data1), np.absolute(data1) == 0),
            np.logical_or(np.isnan(data2), np.absolute(data2) == 0)))

    data1 = data1[in_mask_indices]
    data2 = data2[in_mask_indices]
    
    corr_coeff = np.corrcoef(data1, data2)[0][1]

    return 1-corr_coeff

def trainer(train_loader, model, lr, batch_size, epochs, output_dir, model_name, vox_loss='mse', ngpu=1, seed=42):
    str_lr = "{:.0e}".format(lr)
    # Set random seed for reproducibility
    manualSeed = seed
    print("Random Seed: ", manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    # Decide which device we want to run on
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
    print('Use:', device)

    cuda = True if torch.cuda.is_available() else False

    # Calculate output of image discriminator (PatchGAN)
    if 'pixelGAN' in output_dir:
        patch = (1, 48 // 2 ** 2, 56 // 2 ** 2, 48 // 2 ** 2)
    elif 'imageGAN' in output_dir:
        patch = (1, 1, 1, 1)
    else:
        patch = (1, 48 // 2 ** 4, 56 // 2 ** 4, 48 // 2 ** 4)

    # Initialize generator and discriminator
    generator = model.GeneratorUNet()
    discriminator = model.Discriminator()

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        generator = nn.DataParallel(generator)
        discriminator = nn.DataParallel(discriminator)

    if cuda:
        generator = generator.to(device)
        discriminator = discriminator.to(device)

    # Loss functions
    criterion_GAN = torch.nn.MSELoss()
    criterion_voxelwise = torch.nn.MSELoss()
    criterion_correlation = Cor_CoeLoss

    if vox_loss == 'mse':
        # Loss weight of L1 voxel-wise loss between translated image and real image
        lambda_voxel = 100
        lambda_corr = 0
    elif vox_loss == 'corrcoef':
        # Loss weight of L1 voxel-wise loss between translated image and real image
        lambda_voxel = 0
        lambda_corr = 100
    elif vox_loss == 'corrcoef-mse':
        lambda_voxel = 100
        lambda_corr = 100
        

    # Initialize weights
    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)
    b1 = 0.5
    b2 = 0.999
    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(b1, b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2))

    # Tensor type
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    # ----------
    #  Training
    # ----------
    prev_time = time.time()
    discriminator_update = 'False'

    for epoch in range(epochs):
        for i, batch in enumerate(train_loader):

            # Model inputs
            real_A = Variable(batch[0].type(Tensor))
            real_B = Variable(batch[1].type(Tensor))

            # Adversarial ground truths
            valid = Variable(Tensor(np.ones((real_A.size(0), *patch))), requires_grad=False)
            fake = Variable(Tensor(np.zeros((real_A.size(0), *patch))), requires_grad=False)

            # ---------------------
            #  Train Discriminator, only update every disc_update batches
            # ---------------------
            # Real loss
            fake_B = generator(real_A)
            pred_real = discriminator(real_B, real_A)
            loss_real = criterion_GAN(pred_real, valid)

            # Fake loss
            pred_fake = discriminator(fake_B.detach(), real_A)
            loss_fake = criterion_GAN(pred_fake, fake)
            # Total loss
            loss_D = 0.5 * (loss_real + loss_fake)

            d_real_acu = torch.ge(pred_real.squeeze(), 0.5).float()
            d_fake_acu = torch.le(pred_fake.squeeze(), 0.5).float()
            d_total_acu = torch.mean(torch.cat((d_real_acu, d_fake_acu), 0))

            optimizer_D.zero_grad()
            loss_D.backward()
            optimizer_D.step()
            discriminator_update = 'True'

            # ------------------
            #  Train Generators
            # ------------------
            optimizer_D.zero_grad()
            optimizer_G.zero_grad()

            # GAN loss
            fake_B = generator(real_A)
            pred_fake = discriminator(fake_B, real_A)
            loss_GAN = criterion_GAN(pred_fake, valid)
            
            # Voxel-wise loss
            loss_voxel = criterion_voxelwise(fake_B, real_B)
            loss_corr = criterion_correlation(fake_B, real_B)

            # Total loss
            loss_G = loss_GAN + lambda_voxel * loss_voxel + lambda_corr * loss_corr

            loss_G.backward()

            optimizer_G.step()
            

        # Print log
        print(
            "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f, D accuracy: %f, D update: %s] [G loss: %f, voxel: %f, adv: %f, corr: %f]"
            % (
                epoch,
                epochs,
                i,
                len(train_loader),
                loss_D.item(),
                d_total_acu,
                discriminator_update,
                loss_G.item(),
                loss_voxel.item(),
                loss_GAN.item(),
                loss_corr.item()
            )
        )

        discriminator_update = 'False'
        
        if epoch % 50 == 0 or epoch==epochs-1:
            if torch.cuda.device_count() > 1:
                torch.save(generator.module.state_dict(), f"{output_dir}/{model_name}_b-{batch_size}_lr-{str_lr}_e-{epoch}.pt")
            else:
                torch.save(generator.state_dict(), f"{output_dir}/{model_name}_b-{batch_size}_lr-{str_lr}_e-{epoch}.pt")