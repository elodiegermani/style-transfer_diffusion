'''
This scripts implement functions to train or transfer for style transfer 
on 3D images using conditional DDPM.

Diffusion model is based on DDPM,
https://arxiv.org/abs/2006.11239

The conditioning idea is taken from 'Classifier-Free Diffusion Guidance',
https://arxiv.org/abs/2207.12598
'''

from utils.datasets import ClassifDataset
from torch.backends import cudnn 
import torch
from torch.utils.data import DataLoader
import os
import argparse
from diffusion.c_ddpm import DDPM
import matplotlib.pyplot as plt 
from nilearn import plotting
import numpy as np 
import pandas as pd
import nibabel as nib
from nilearn.plotting.cm import _cmap_d as nilearn_cmaps
import importlib
import sys 


def train(config):

    if not os.path.isdir(config.sample_dir):
        os.mkdir(config.sample_dir)
    if not os.path.isdir(config.model_save_dir):
        os.mkdir(config.model_save_dir)

    ddpm = DDPM(config)

    # Data loader. 
    dataset_file = f'{config.data_dir}/{config.mode}-{config.dataset}.csv'

    dataset = ClassifDataset(
        dataset_file, 
        config.labels)

    print(f'Dataset {config.dataset}: \n {len(dataset)} images.')


    loader = DataLoader(
        dataset, 
        batch_size=config.batch_size, 
        shuffle=True, 
        )

    optim = torch.optim.Adam(
        ddpm.parameters(), 
        lr=config.lrate
        )

    for ep in range(config.n_epoch):

        print(f'Epoch {ep}')

        ddpm.train()

        # linear lrate decay
        optim.param_groups[0]['lr'] = config.lrate * (1 - ep / config.n_epoch)

        loss_ema = None

        for i, (x, c) in enumerate(loader):

            optim.zero_grad()

            x = x.to(ddpm.device)
            c = c.to(ddpm.device)

            loss = ddpm(x.float(), c.float())
            loss.backward()

            if loss_ema is None:
                loss_ema = loss.item()
            else:
                loss_ema = 0.95 * loss_ema + 0.05 * loss.item()

            optim.step()

        print('Loss:', loss_ema)

        if ep%10==0:
            torch.save(ddpm.state_dict(), config.model_save_dir + f"/c_ddpm_{ep}.pth")
            
def transfer(config):
    ddpm = DDPM(config)
    ddpm.load_state_dict(
        torch.load(
            config.model_save_dir + f"/model_{config.test_iter}.pth", 
            map_location=ddpm.device
            )
        )

    # Data loader. 
    dataset_file = f'{config.data_dir}/test-{config.dataset}.csv'

    dataset = ClassifDataset(
        dataset_file, 
        config.labels)

    print(f'Dataset {config.dataset}: \n {len(dataset)} images.')


    source_loader = DataLoader(
        dataset, 
        batch_size=1, 
        shuffle=False, 
        )

    df_metrics = pd.DataFrame(
            columns = ['orig_label', 'target_label', 'orig-target', 'orig-gen', 'gen-target', 'w']
            )

    for n, (x, c) in enumerate(source_loader):

        ddpm.eval()

        with torch.no_grad():

            for i in range(config.n_classes):
                x_r, c_r = dataset[n//config.n_classes*config.n_classes+i]

                c_t = c_r.view(-1, c_r.shape[0])

                for w_i, w in enumerate(config.ws_test):

                    x_gen = ddpm.transfer(
                        x, 
                        c_t, 
                        guide_w=w
                        )

                    affine = np.array([[   4.,    0.,    0.,  -98.],
                                       [   0.,    4.,    0., -134.],
                                       [   0.,    0.,    4.,  -72.],
                                       [   0.,    0.,    0.,    1.]])

                    img_xgen = nib.Nifti1Image(
                        np.array(
                            x_gen.detach().cpu()
                            )[0,0,:,:,:], 
                        affine
                        )

                    img_xreal = nib.Nifti1Image(
                        np.array(
                            x_r.detach().cpu()
                            )[0,:,:,:], 
                        affine
                        )

                    img_xsrc = nib.Nifti1Image(
                        np.array(
                            x.detach().cpu()
                            )[0,0,:,:,:], 
                        affine
                        )

                    p_source = dataset.label_list[torch.argmax(c, dim=1)[0]]
                    p_target = dataset.label_list[torch.argmax(c_t, dim=1)[0]]

                    nib.save(img_xgen, f'{config.sample_dir}/gen_img-{n}_w{w}_orig-{p_source}_target-{p_target}.nii.gz')
                    nib.save(img_xreal, f'{config.sample_dir}/trg_img-{n}_w{w}_orig-{p_source}_target-{p_target}.nii.gz')
                    nib.save(img_xsrc, f'{config.sample_dir}/src_img-{n}_w{w}_orig-{p_source}_target-{p_target}.nii.gz')