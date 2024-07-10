import sys
from torch.utils.data import DataLoader
from os.path import join as opj
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import random
from nilearn import plotting
import nibabel as nib 
import os 

def tester(valid_dataset, generator, sample_dir, subset_A, subset_B):
    affine = nib.load(valid_dataset.data_A[1]).affine 
    header = nib.load(valid_dataset.data_A[1]).header

    data_loader = DataLoader(
                    valid_dataset, 
                    batch_size=1,
                    shuffle=False)

    generator.eval()

    device= 'cpu'
    with torch.no_grad():
        for i, data in enumerate(data_loader):
            imgs = data[0].float().to(device) # Original source image 
            out = generator(imgs).cpu() # Translated image

            source_img = data[0].float().cpu()
            target_img = data[1].float().cpu()
            generated_img = out.float() * np.array(target_img != 0).astype('float')

            #generated_img = utils.un_normalize(generated_img, target_img)
            nib.save(nib.Nifti1Image(source_img, affine, header), opj(sample_dir, f'src_img-{i}_orig-{subset_A}_target-{subset_B}.nii.gz'))
            nib.save(nib.Nifti1Image(target_img, affine, header), opj(sample_dir, f'trg_img-{i}_orig-{subset_A}_target-{subset_B}.nii.gz'))
            nib.save(nib.Nifti1Image(generated_img, affine, header), opj(sample_dir, f'gen_img-{i}_orig-{subset_A}_target-{subset_B}.nii.gz'))