'''
This scripts implement functions to train and test a classifier 3D CNN. 
'''

from models import classifier as md 
from classification import train 
from utils import datasets, feature_visualization
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader
import sys
import os
from sklearn import metrics

def train(config):
	# Parameters:
	### TO MODIFY
	lr=config.lrate
	str_lr = "{:.0e}".format(lr)
	batch_size=config.batch_size
	epochs=config.n_epoch
	out_dir = config.model_save_dir
	data_dir = config.data_dir

	# Datasets
	label_file = f'{config.data_dir}/train-{config.dataset}.csv'
	label_column = config.labels
	train_dataset = datasets.ClassifDataset(
		label_file, 
		label_column
		)

	label_file = f'{config.data_dir}/test-{config.dataset}.csv'
	valid_dataset = datasets.ClassifDataset(
		label_file, 
	    label_column
	    )

	# Reproducibility constraints
	random_seed=42
	np.random.seed(random_seed)
	torch.manual_seed(random_seed)
	np.random.seed(random_seed)
	torch.random.manual_seed(random_seed)
	torch.cuda.manual_seed(random_seed)

	torch.backends.cudnn.deterministic = True 
	torch.backends.cudnn.benchmark = False

	# Create output dir to save models
	if not os.path.isdir(out_dir): 
	    os.mkdir(out_dir)

	# GPU or CPU
	if torch.cuda.is_available():
	    device = torch.device("cuda")
	    torch.cuda.manual_seed(random_seed)
	    print('Using GPU.')
	else:
	    device = "cpu"
	    print('Using CPU.')

	# Loss to use
	distance = nn.CrossEntropyLoss()

	# Model
	model = md.Classifier3D(
		n_class = len(train_dataset.label_list)
		)

	print('Model:', model)
	model = model.to(device)
	
	# Optimizer
	print(f'Optimizer: ADAM, lr={lr}')
	optimizer = torch.optim.Adam(model.parameters(), lr=lr)  

	# DataLoader
	train_dataset = DataLoader(
		train_dataset, 
		batch_size=batch_size, 
		shuffle=True
		)

	valid_dataset = DataLoader(
		valid_dataset, 
		batch_size=batch_size
		)

	# START TRAINING
	print('Start training...')
	training_loss = []
	validation_loss = []

	for epoch in range(epochs):
		# TRAINING LOOP
	    current_training_loss = train.train(
	    	model, 
	    	train_dataset, 
	    	distance, 
	    	optimizer, 
	    	device
	    	)

	    training_loss.append(
	    	current_training_loss
	    	)

	    # VALIDATION LOOP
	    current_validation_loss = train.validate(
	    	model, 
	    	valid_dataset, 
	    	distance, 
	    	device
	    	)

	    validation_loss.append(
	    	current_validation_loss
	    	)
	    
	    print('epoch [{}/{}], loss:{:.4f}, validation:{:.4f}'.format(
	    	epoch + 1, epochs, current_training_loss,
	        current_validation_loss
	        )
	    )
	    
	    if device != 'cpu':
	        if device.type == 'cuda':
	            torch.cuda.empty_cache()

	    if epoch%10==0 or epoch == epochs:
	        torch.save(
	        	model.state_dict(),  
	        	f"{out_dir}/classifier_b-{batch_size}_lr-{str_lr}_epochs_{epoch}.pth") 

	print('Training ended')

def test(config):
	label_file = f'{config.data_dir}/test-{config.dataset}.csv'
	valid_dataset = datasets.ClassifDataset(
		label_file, 
	    config.labels
	    )

	classifier = md.Classifier3D(
		n_class = len(valid_dataset.label_list)
		)

	classifier.load_state_dict(
            torch.load(
                config.model_param, 
                map_location='cpu'
                )
            )

	classifier.eval()

	correlation = feature_visualization.get_correlation_features(classifier, valid_dataset)

	return correlation
