'''
This scripts launch train or transfer for style transfer on 3D images and 
training of the classifier used for feature extraction.

Diffusion model is based on DDPM,
https://arxiv.org/abs/2006.11239

The conditioning idea is taken from 'Classifier-Free Diffusion Guidance',
https://arxiv.org/abs/2207.12598
'''
import os
import argparse 
from scripts import main_cc_ddpm, main_c_ddpm, main_classifier

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, 
        help='model to train, one of cc_ddpm, c_ddpm, classifier', default='cc_ddpm')
    parser.add_argument('--data_dir', type=str, 
        help = 'where data csv files are stored', default='data')
    parser.add_argument('--dataset', type=str, 
        help = 'name of the dataset to use', default='global_dataset')
    parser.add_argument('--labels', type=str, 
        help='conditions for generation', default='pipelines')
    parser.add_argument('--sample_dir', type=str, 
        help='sampling directory', default = 'results/samples')
    parser.add_argument('--model_save_dir', type=str, 
        help='save directory', default = 'results/models')
    parser.add_argument('--mode', type=str, help='train or transfer', 
        default='train', choices=['train', 'transfer', 'test'])
    parser.add_argument('--batch_size', type=int, 
        default=32, help='mini-batch size')
    parser.add_argument('--n_epoch', type=int, 
        default=500, help='number of total iterations')
    parser.add_argument('--lrate', type=float, 
        default=1e-4, help='learning rate')
    parser.add_argument('--n_feat', type=int, 
        default=64, help='number of features')
    parser.add_argument('--n_classes', type=int, 
        default=24, help='number of classes')
    parser.add_argument('--beta', type=tuple, 
        default=(1e-4, 0.02), help='beta')
    parser.add_argument('--n_T', type=int, 
        default=500, help='number T: nb of timesteps')
    parser.add_argument('--n_C', type=int, 
        default=10, help='number C: nb of images for sampling')
    parser.add_argument('--sampling', type=str, 
        default='random', help='how to sample target img, one of random, knn or kmeans')
    parser.add_argument('--drop_prob', type=float, 
        default=0.1, help='probability drop for unconditional ddpm')
    parser.add_argument('--ws_test', type=int, 
        default=[0.5], help='weight strengh for sampling')
    parser.add_argument('--test_iter', type=int, 
        default=100, help='epoch of model to test')
    parser.add_argument('--model_param', type=str, 
        default='./results/models/classifier_b-64_lr-1e-04_epochs_140.pth', 
        help='model to use for feature extraction')

    config = parser.parse_args()

    print(config)

    if config.model == 'c_ddpm':
        if config.mode == 'train':
            main_c_ddpm.train(config)

        elif config.mode == 'transfer':
            main_c_ddpm.transfer(config)

    elif config.model == 'cc_ddpm':
        if config.mode == 'train':
            main_cc_ddpm.train(config)

        elif config.mode == 'transfer':
            main_cc_ddpm.transfer(config)

    elif config.model == 'classifier':
        if config.mode == 'train':
            main_classifier.train(config)
        elif config.mode == 'test':
            main_classifier.test(config)