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
from scripts import main_cc_ddpm, main_c_ddpm, main_classifier, main_stargan

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, 
        help='model to train, one of cc_ddpm, c_ddpm, classifier, stargan, pix2pix or cycleGAN', default='cc_ddpm')
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
    parser.add_argument('--c_dim', type=int, 
        default=5, help='dimension of domain labels (1st dataset)')
    parser.add_argument('--image_size', type=int, 
        default=56, help='image resolution')
    parser.add_argument('--g_conv_dim', type=int, 
        default=64, help='number of conv filters in the first layer of G')
    parser.add_argument('--d_conv_dim', type=int, 
        default=64, help='number of conv filters in the first layer of D')
    parser.add_argument('--g_repeat_num', type=int, 
        default=6, help='number of residual blocks in G')
    parser.add_argument('--d_repeat_num', type=int, 
        default=4, help='number of strided conv layers in D')
    parser.add_argument('--lambda_cls', type=float, 
        default=1, help='weight for domain classification loss')
    parser.add_argument('--lambda_rec', type=float, 
        default=10, help='weight for reconstruction loss')
    parser.add_argument('--lambda_gp', type=float, 
        default=10, help='weight for gradient penalty')
    parser.add_argument('--num_iters', type=int, 
        default=200000, help='number of total iterations for training D')
    parser.add_argument('--num_iters_decay', type=int, 
        default=100000, help='number of iterations for decaying lr')
    parser.add_argument('--g_lr', type=float, 
        default=0.0001, help='learning rate for G')
    parser.add_argument('--d_lr', type=float, 
        default=0.0001, help='learning rate for D')
    parser.add_argument('--n_critic', type=int, 
        default=5, help='number of D updates per each G update')
    parser.add_argument('--beta1', type=float, 
        default=0.5, help='beta1 for Adam optimizer')
    parser.add_argument('--beta2', type=float, 
        default=0.999, help='beta2 for Adam optimizer')
    parser.add_argument('--resume_iters', type=int, 
        default=None, help='resume training from this step')
    parser.add_argument('--log_step', 
        type=int, default=10)
    parser.add_argument('--model_save_step', 
        type=int, default=10000)
    parser.add_argument('--lr_update_step', 
        type=int, default=10000)

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

    elif config.model == 'stargan':
        main_stargan.main(config)

    elif config.model == 'cycleGAN':
        main_cyclegan.main(config)

    elif config.model == 'pix2pix':
        main_pix2pix.main(config)

    elif config.model == 'classifier':
        if config.mode == 'train':
            main_classifier.train(config)
        elif config.mode == 'test':
            main_classifier.test(config)