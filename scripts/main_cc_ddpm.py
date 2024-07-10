'''
This scripts implement functions to train or transfer for style transfer 
on 3D images using classifier-conditional DDPM.

Diffusion model is based on DDPM,
https://arxiv.org/abs/2006.11239

The conditioning idea is taken from 'Classifier-Free Diffusion Guidance',
https://arxiv.org/abs/2207.12598

Conditioning is performed in the latent space of a classifier using multiple target. 
'''

from utils.datasets import ClassifDataset
from torch.backends import cudnn 
import torch
from torch.utils.data import DataLoader
import os
import argparse
from diffusion.cc_ddpm import DDPM
import numpy as np 
import pandas as pd
import random
import nibabel as nib 
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans

def train(config):
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

            loss = ddpm(x.float())
            loss.backward()

            if loss_ema is None:
                loss_ema = loss.item()
            else:
                loss_ema = 0.95 * loss_ema + 0.05 * loss.item()

            optim.step()

        print('Loss:', loss_ema)

        if ep %10 == 0 or ep == config.n_epoch:
            torch.save(ddpm.state_dict(), config.model_save_dir + f"/cc_ddpm_{ep}.pth")

            

def transfer(config):
    if not os.path.isdir(config.sample_dir):
        os.mkdir(config.sample_dir)
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

    dataset_file = f'{config.data_dir}/train-{config.dataset}.csv'
    train_dataset = ClassifDataset(
        dataset_file, 
        config.labels)

    for n, (x, c) in enumerate(source_loader):

        ddpm.eval()

        with torch.no_grad():
            for w in config.ws_test:
                for i in range(config.n_classes):

                    c_idx = dataset.label_list[torch.argmax(c, dim=1)[0]]
                    c_t_idx = dataset.label_list[i]

                    class_idx = [cl for cl in range(len(train_dataset.get_original_labels())) if train_dataset.get_original_labels()[cl]==c_t_idx]

                    if config.sampling == 'random':
                        i_t_list = random.sample(class_idx, config.n_C)
                        x_t_list = []
                        for i_t in i_t_list:

                            x_t, c_t = train_dataset[i_t]
                            x_t_list.append(x_t)

                    elif config.sampling == 'knn': 
                        i_t_list = class_idx

                        x_t_list = []
                        for i_t in i_t_list:

                            x_t, c_t = train_dataset[i_t]
                            x_t_list.append(torch.flatten(x_t))

                        knn = NearestNeighbors(
                            n_neighbors=config.n_C)
                        knn.fit(
                            np.array(
                                x_t_list))
                        idx_t_list = knn.kneighbors(
                            torch.flatten(x, start_dim=1), 
                            config.n_C, 
                            return_distance=False)[0]

                        x_t_list = []
                        for i_t in idx_t_list:
                            x_t, c_t = train_dataset[i_t_list[i_t]]
                            x_t_list.append(x_t)

                    elif config.sampling == 'kmeans':
                        i_t_list = class_idx

                        x_t_list = []
                        for i_t in i_t_list:

                            x_t, c_t = train_dataset[i_t]
                            x_t_list.append(torch.flatten(x_t))

                        kmeans = KMeans(n_clusters=config.n_C, random_state=0, n_init="auto")
                        kmeans.fit(np.array(x_t_list))

                        x_t_list = [torch.tensor(center.reshape(1,48,56,48)) for center in kmeans.cluster_centers_]


                    x_gen = ddpm.transfer(
                    x, x_t_list, w
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

                    x_r, c_r = dataset[n//config.n_classes*config.n_classes+i]

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

                    nib.save(img_xgen, f'{config.sample_dir}/gen_img-{n}_orig-{c_idx}_target-{c_t_idx}.nii.gz')
                    nib.save(img_xreal, f'{config.sample_dir}/trg_img-{n}_orig-{c_idx}_target-{c_t_idx}.nii.gz')
                    nib.save(img_xsrc, f'{config.sample_dir}/src_img-{n}_orig-{c_idx}_target-{c_t_idx}.nii.gz')