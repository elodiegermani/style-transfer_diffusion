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

def trainer(train_loader, md, lr, batch_size, epochs, output_dir, model_name, seed=42):
    str_lr = "{:.0e}".format(lr)
    model = md.CycleGAN(lr=lr)
    # Set random seed for reproducibility
    manualSeed = seed
    print("Random Seed: ", manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    #train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)  # Here are then fed to the network with a defined batch size
    for epoch in range(epochs):
        for i, data in enumerate(train_loader):
            model.set_input(data)
            model.optimize_parameters()

        losses = model.get_current_losses()
        
        print(epoch, losses)
        torch.cuda.empty_cache()

        if epoch % 10 == 0:
            torch.save(model, opj(output_dir, f'{model_name}_b-{batch_size}_lr-{str_lr}_e-{epoch}.pt'))











