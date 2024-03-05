# Mitigating analytical variability through style transfer

This repository contains scripts to train and evaluate a conditional diffusion models to perform unsupervised image-to-image transition using conditioning on multiple target images on the latent space of a classifier. 

Selection of target images is implemented using random selection, K-means clustering and KNN identification. 

Conditional diffusion is inspired from ['Classifier-Free Diffusion Guidance'](https://arxiv.org/abs/2207.12598). 

<p align = "center">
<img width="400" src="results/transfers.gif"/img>
</p>
<p align = "center">
<br>Figure 1.</br> Generated images for all transfers using CC-DDPM.
</p>


|                                        | IS     | <td colspan=2>fsl-1 to spm-0 | <td colspan=2>spm-0 to fsl-1 | <td colspan=2>fsl-1 to spm-1 | <td colspan=2>fsl-1 to fsl-0 |
|                                        | IS     | Corr.  | PSNR   | Corr.  | PSNR   | Corr.  | PSNR   | Corr.  | PSNR   |
|----------------------------------------|--------|--------|--------|--------|--------|--------|--------|--------|--------|
| Initial                                | 3.69 | 76.2 | 78.2 | 76.2 | 78.2 | 82.6 | 81.3 | 91.0 | 83.9 |
| One-hot (Ho & al., 2022) | 3.66   | 83.8   | 77.2   | 75.0   | 79.4   | 78.7   | 77.7   | 81.1   | 79.5   |
| N=1 (Preechakul & al., 2022)  | 3.70   | 85.5   | 79.0   | 77.8   | 80.0   | 79.9   | 78.0   | 82.8   | 80.2   |
| StarGAN (Choi & al., 2018)      | 3.63   | **90.5** | **81.9** | **87.5** | **84.2** | **87.6** | **81.8** | **91.5** | **85.0** |
| CCDDPM                                 | **3.93** | 86.1   | 79.4   | 79.0   | 80.7 | 81.2   | 78.9   | 84.1   | 80.6   |
| N=5                                    | 3.86 | 86.4   | 79.8 | 78.7   | 80.6   | 81.2   | 79.4 | 84.5 | 80.9 |
| N=20                                   | 86.1   | 79.5   | 79.2   | 80.7 | 81.3   | 79.2   | 83.9   | 80.9   |
| N=5, random                            | 3.89 | 86.5   | 79.4   | 79.1   | 80.4   | 82.0 | 79.2   | 84.2   | 80.2   |
| N=10, random                           | 3.86 | 86.5   | 79.2   | 79.0   | 80.2   | 81.8   | 79.4 | 84.3   | 80.8   |
| N=20, random                           | 3.85 | 86.7 | 79.1   | 79.3 | 80.6   | 81.5   | 79.4 | 84.4   | 80.7   |
| N=10, KNN                              | 3.75   | 84.9   | 79.3   | 78.7   | 80.0   | 81.6   | 79.1   | 83.6   | 80.7   |

<p align = "center">
<br>Table 1.</br> Performance associated with four transfers. IS means ”Inception Score” across all transfers. Pearson’s correlation (%) and Peak Signal to Noise Ration (PSNR) computed between generated and ground-truth target image for 20 images per transfer. Initial represents the metrics between the source image (before transfer) and the ground-truth target image. Boldface marks the top model. </p>

<p align = "center">
<img width="400" src="results/figures/visualization.png"/img>
</p>
<p align = "center">
<br>Figure 2.</br> Generated images for two transfer and different competitors: conditioning with one-hot encoding (Ho & al., 2022), with a classifier and N=1 (Preechakul & al., 2022),  starGAN (Choi & al., 2018) and CCDPM.
</p>

## How to reproduce ? 

If you use pre-trained models, for each command used to evaluate performance, change `--model_param` to the path of the pre-trained classifier and `--model_save_dir` to the path of the directory containing the pre-trained diffusion models. 

### Classifier

#### Train
```bash
python3.10 -u main.py --model classifier --data_dir data --dataset dataset_rh_4classes --labels pipelines --model_save_dir results/models --batch_size 64 --lrate 1e-4 --n_epoch 150
```

#### Evaluate 

```bash 
python3.10 -u main.py --model classifier --data_dir data --dataset dataset_rh_4classes --labels pipelines --mode test --model_param ./results/models/classifier_b-64_lr-1e-04_epochs_150.pth
```

### Diffusion models 
#### Train 

```bash
python3.10 -u main.py --model cc_ddpm --mode train --dataset dataset_rh_4classes --labels pipelines --model_save_dir results/models --batch_size 8 --lrate 1e-4 --n_epoch 200 --n_classes 4 --sample_dir results/samples
```

#### Transfer

```bash
python3.10 -u main.py --model cc_ddpm --mode transfer --dataset dataset_rh_4classes --labels pipelines --model_save_dir results/models --test_iter 200 --n_classes 4 --sample_dir results/samples
```

### StarGAN

####  Train
```bash
python3.10 -u main.py --model stargan --mode train --dataset dataset_rh_4classes --labels pipelines --image_size 56 --c_dim 4 --batch_size 16 --data_dir data --sample_dir results/samples --model_save_dir results/models
```

#### Test
```bash
python3.10 -u main.py --model stargan --mode test --dataset dataset_rh_4classes --labels pipelines --image_size 56 --c_dim 4 --batch_size 1 --data_dir data --sample_dir results/samples --model_save_dir results/models --test_iters 100000
```
