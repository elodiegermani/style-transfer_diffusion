import os
import argparse
from stargan.solver import Solver
from utils.datasets import ClassifDataset
from torch.backends import cudnn
from torch.utils.data import DataLoader

def str2bool(v):
    return v.lower() in ('true')

def main(config):
    # For fast training.
    cudnn.benchmark = True

    # Create directories if not exist.
    if not os.path.exists(config.model_save_dir):
        os.makedirs(config.model_save_dir)
    if not os.path.exists(config.sample_dir):
        os.makedirs(config.sample_dir)

    # Data loader. 
    dataset_file = f'{config.data_dir}/{config.mode}-{config.dataset}.csv'

    dataset = ClassifDataset(
        dataset_file, 
        config.labels)

    print(f'Dataset {config.dataset}: \n {len(dataset)} images.')

    data_loader = DataLoader(
        dataset, 
        batch_size=config.batch_size,
        shuffle=True)
    
    if config.mode == 'test':
        
        data_loader = DataLoader(
        dataset, 
        batch_size=config.batch_size,
        shuffle=False)

    # Solver for training and testing StarGAN.
    solver = Solver(data_loader, config)

    if config.mode == 'train':
        solver.train()
    elif config.mode == 'test':
        solver.test(dataset)