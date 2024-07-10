#!/bin/bash
#SBATCH --job-name=tr-cyclegan-rh # nom du job
#SBATCH --ntasks=1                   # number of MP tasks
#SBATCH --ntasks-per-node=1          # number of MPI tasks per node
#SBATCH --partition=gpu_p13
#SBATCH --gres=gpu:1                 # number of GPUs per node
#SBATCH --cpus-per-task=16           # number of cores per tasks
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --distribution=block:block   # we pin the tasks on contiguous cores
#SBATCH --time=19:00:00              # maximum execution time (HH:MM:SS)
#SBATCH --output=tr-cyclegan-rh%j.out # output file name
#SBATCH --error=tr-cyclegan-rh%j.err  # error file name

source /gpfswork/rech/gft/umh25bv/miniconda3/bin/activate /gpfswork/rech/gft/umh25bv/miniconda3/envs/workEnv

# Feature extractor 
# /gpfswork/rech/gft/umh25bv/miniconda3/envs/workEnv/bin/python3 -u /gpfswork/rech/gft/umh25bv/style-transfer_diffusion/main.py \
# --model feature_extractor --data_dir data \
# --dataset dataset_rh_4classes-jeanzay --labels pipelines \
# --model_save_dir feature_extractor/models --batch_size 64 \
# --lrate 1e-4 --n_epoch 150

# Pix2Pix / CycleGAN
/gpfswork/rech/gft/umh25bv/miniconda3/envs/workEnv/bin/python3 -u /gpfswork/rech/gft/umh25bv/style-transfer_diffusion/main.py \
--model cycleGAN --mode train --dataset dataset_rh_4classes-jeanzay \
--labels pipelines --image_size 56 --c_dim 4 --batch_size 16 \
--data_dir data --sample_dir results/samples/cycleGAN-rh \
--model_save_dir results/models/cycleGAN-rh

# StarGAN 
## Train
# /gpfswork/rech/gft/umh25bv/miniconda3/envs/workEnv/bin/python3 -u /gpfswork/rech/gft/umh25bv/style-transfer_diffusion/main.py \
# --model stargan --mode train --dataset dataset_rf_4classes-jeanzay \
# --labels pipelines --image_size 56 --c_dim 4 --batch_size 16 \
# --data_dir data --sample_dir results/samples/stargan-rf \
# --model_save_dir results/models/stargan-rf

# Test
# /gpfswork/rech/gft/umh25bv/miniconda3/envs/workEnv/bin/python3 -u /gpfswork/rech/gft/umh25bv/style-transfer_diffusion/main.py \
# --model stargan --mode test --dataset dataset_rh_4classes-jeanzay \
# --labels pipelines --image_size 56 --c_dim 4 --batch_size 1 \
# --data_dir data --sample_dir results/samples/stargan-rf--rh \
# --model_save_dir results/models/stargan-rf --test_iter 70000

#C-DDPM
## Train

## Test 
# /gpfswork/rech/gft/umh25bv/miniconda3/envs/workEnv/bin/python3 -u /gpfswork/rech/gft/umh25bv/style_transfer/main.py \
# --model c_ddpm --mode transfer --dataset dataset_rh_4classes-jeanzay \
# --labels pipelines    --batch_size 1 --data_dir data --sample_dir c_ddpm/samples \
# --model_save_dir c_ddpm/models --test_iter 90 --n_classes 4

# CC-DDPM
## Train
# /gpfswork/rech/gft/umh25bv/miniconda3/envs/workEnv/bin/python3 -u /gpfswork/rech/gft/umh25bv/style_transfer/main.py \
# --model cc_ddpm --mode train --dataset dataset_rh_4classes-jeanzay \
# --labels pipelines --model_save_dir cc_ddpm/models \
# --batch_size 8 --lrate 1e-4 --n_epoch 200 --n_classes 4 \
# --sample_dir cc_ddpm/samples

## Test
#  /gpfswork/rech/gft/umh25bv/miniconda3/envs/workEnv/bin/python3 -u /gpfswork/rech/gft/umh25bv/style_transfer/main.py \
# --model cc_ddpm --mode transfer --dataset dataset_rh_4classes-jeanzay \
# --labels pipelines --model_save_dir cc_ddpm/models \
# --batch_size 1 --n_classes 4 --sampling kmeans --n_C 5 \
# --sample_dir cc_ddpm/samples-kmeans --test_iter 190 

# CC-DDPM-S
## Train
# /gpfswork/rech/gft/umh25bv/miniconda3/envs/workEnv/bin/python3 -u /gpfswork/rech/gft/umh25bv/style_transfer/main.py \
# --model cc_ddpm_s --mode train --dataset dataset_rh_4classes-jeanzay \
# --labels pipelines --model_save_dir cc_ddpm_s/models \
# --batch_size 8 --lrate 1e-4 --n_epoch 200 --n_classes 4 \
# --sample_dir cc_ddpm_s/samples

## Test
#  /gpfswork/rech/gft/umh25bv/miniconda3/envs/workEnv/bin/python3 -u /gpfswork/rech/gft/umh25bv/style_transfer/main.py \
# --model cc_ddpm_s --mode transfer --dataset dataset_rh_4classes-jeanzay \
# --labels pipelines --model_save_dir cc_ddpm_s/models \
# --batch_size 1 --n_classes 4 \
# --sample_dir cc_ddpm_s/samples --test_iter 190 
