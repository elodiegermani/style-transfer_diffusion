import os
import argparse
from cycleGAN.cycleGAN_trainer import trainer
from cycleGAN.cycleGAN_tester import tester
from utils.datasets import ClassifDataset
from torch.backends import cudnn
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import importlib

def main(config):
    # For fast training.
    cudnn.benchmark = True

    # Create directories if not exist.
    if not os.path.exists(config.model_save_dir):
        os.makedirs(config.model_save_dir)
    if not os.path.exists(config.sample_dir):
        os.makedirs(config.sample_dir)

    if config.mode == 'train':
        # Data loader. 
        dataset_file = f'{config.data_dir}/{config.mode}-{config.dataset}.csv'
        df_pipelines = pd.read_csv(dataset_file)
        subset_list = np.unique(df_pipelines['pipelines']).tolist()

        for subset_A in subset_list:
            for subset_B in subset_list:

                dataset = UnpairedImageDataset(subset_A, subset_B, dataset_file, contrast_list=['right-hand'])

                print(f'Dataset {config.dataset}: \n {len(dataset)} images.')

                data_loader = DataLoader(
                    dataset, 
                    batch_size=config.batch_size,
                    shuffle=True)

                package = 'models.' + config.model
                md = importlib.import_module(package)

                model_name = f'model-CycleGAN_{subset_A}-to-{subset_B}'

                trainer(data_loader, md, config.lrate, config.batch_size, config.n_epoch, config.model_save_dir, model_name)

    elif config.mode == 'transfer':
        # Data loader. 
        dataset_file = f'{config.data_dir}/{config.mode}-{config.dataset}.csv'
        df_pipelines = pd.read_csv(dataset_file)
        subset_list = np.unique(df_pipelines['pipelines']).tolist()


        for subset_A in subset_list:
            for subset_B in subset_list:

                valid_dataset = ds.PairedImageDataset(subset_A, subset_B, dataset_file, contrast_list=['right-hand'])
                print('Number of images in TEST:', len(valid_dataset.data_B))

                str_lr = "{:.0e}".format(config.lrate)

                parameter_file = f"{config.model_save_dir}/model-CycleGAN_{subset_A}-to-{subset_B}_b-{config.batch_size}_lr-{str_lr}_e-{config.n_epoch}.pt"

                package = 'models.' + config.model
                md = importlib.import_module(package)
        
                cyclegan = torch.load(parameter_file, map_location="cpu")
                generator = cyclegan.netG_A

                tester(valid_dataset, generator, config.sample_dir, subset_A, subset_B)







