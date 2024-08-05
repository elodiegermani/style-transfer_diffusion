import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from nilearn.plotting.cm import _cmap_d as nilearn_cmaps
import matplotlib.pyplot as plt 
import numpy as np
from nilearn import datasets, plotting
import nibabel as nib
import pandas as pd 
from utils.metrics import get_correlation
from models.classifier import Classifier3D

def visualize_features(parameters_file, dataset, classe, print_title=True):
	model = Classifier3D(
		n_class=4
		)

	state_dict = torch.load(parameters_file, map_location='cpu')

	model.load_state_dict(state_dict)

	# we will save the conv layer weights in this list
	model_weights =[]
	#we will save the 49 conv layers in this list
	conv_layers = []
	# get all the model children as list
	model_children = list(model.children())
	#counter to keep count of the conv layers
	counter = 0
	#append all the conv layers and their respective wights to the list
	for i in range(len(model_children)):
		if type(model_children[i]) == nn.Conv3d:
			counter+=1
			model_weights.append(model_children[i].weight)
			conv_layers.append(model_children[i])
		elif type(model_children[i]) == nn.Sequential:
			for j in range(len(model_children[i])):
				for child in model_children[i][j].children():
					if type(child) == nn.Conv3d:
						counter+=1
						model_weights.append(child.weight)
						conv_layers.append(child)
	#print(f"Total convolution layers: {counter}")
	#print(conv_layers)

	all_img = [[] for i in range(5)]
	all_img_full = [[] for i in range(5)]

	image_loader = DataLoader(dataset, batch_size = 1, shuffle=False)
	for i, data in enumerate(image_loader):
		if dataset.get_original_labels()[i] == classe:
			image = data[0].float()

			outputs = []
			names = []
			for layer in conv_layers[0:]:
				image = layer(image)
				outputs.append(image)
				names.append(str(layer))
			#print(len(outputs))
			#print feature_maps
			#for feature_map in outputs:
				#print(feature_map.shape)

			processed = []
			for feature_map in outputs:
				feature_map = feature_map.squeeze(0)
				gray_scale = torch.sum(feature_map,0)
				gray_scale = gray_scale / feature_map.shape[0]
				processed.append(gray_scale.data.cpu().numpy())
			#for fm in processed:
				#print(fm.shape)

			all_img[0].append(data[0].cpu().numpy()[0,0,:,:,int(data[0].cpu().numpy().shape[2]/2)])
			all_img_full[0].append(data[0].cpu().numpy()[0,0,:,:,:])

			for i in range(1, len(all_img)):
				all_img[i].append(processed[i-1][:,:,int(processed[i-1].shape[2]/2)])
				all_img_full[i].append(processed[i-1][:,:,:])

	fig = plt.figure(figsize=(25, 5))
	if print_title:
		fig.suptitle(f'{classe}', fontsize=28, weight='bold')
	for i in range(len(all_img)):
		a = fig.add_subplot(1,5, i+1)
		mean_img = np.mean(all_img[i], axis=0)

		imgplot = plt.imshow(mean_img, cmap = nilearn_cmaps['cold_hot'], vmin=0, vmax=255)
		#plt.colorbar()
		a.axis("off")
		if print_title:
			if i == 0:
				a.set_title(f'ORIGINAL IMAGE', fontsize=20, weight='bold')
			else:
				a.set_title(f'LAYER {i}', fontsize=20, weight='bold')
	plt.savefig(f'./results/figures/mean_features_{classe}.png')
	plt.show()


def compute_features(model, dataset, classe):

	# we will save the conv layer weights in this list
	model_weights =[]
	#we will save the 49 conv layers in this list
	conv_layers = []
	# get all the model children as list
	model_children = list(model.children())
	#counter to keep count of the conv layers
	counter = 0
	#append all the conv layers and their respective wights to the list
	for i in range(len(model_children)):
		if type(model_children[i]) == nn.Conv3d:
			counter+=1
			model_weights.append(model_children[i].weight)
			conv_layers.append(model_children[i])
		elif type(model_children[i]) == nn.Sequential:
			for j in range(len(model_children[i])):
				for child in model_children[i][j].children():
					if type(child) == nn.Conv3d:
						counter+=1
						model_weights.append(child.weight)
						conv_layers.append(child)
	#print(f"Total convolution layers: {counter}")
	#print(conv_layers)

	all_img = [[] for i in range(5)]
	all_img_full = [[] for i in range(5)]

	image_loader = DataLoader(dataset, batch_size = 1, shuffle=False)
	for i, data in enumerate(image_loader):
		if dataset.get_original_labels()[i] == classe:
			image = data[0].float()

			outputs = []
			names = []
			for layer in conv_layers[0:]:
				image = layer(image)
				outputs.append(image)
				names.append(str(layer))
			#print(len(outputs))
			#print feature_maps
			#for feature_map in outputs:
				#print(feature_map.shape)

			processed = []
			for feature_map in outputs:
				feature_map = feature_map.squeeze(0)
				gray_scale = torch.sum(feature_map,0)
				gray_scale = gray_scale / feature_map.shape[0]
				processed.append(gray_scale.data.cpu().numpy())
			#for fm in processed:
				#print(fm.shape)

			all_img[0].append(data[0].cpu().numpy()[0,0,:,:,int(data[0].cpu().numpy().shape[2]/2)])
			all_img_full[0].append(data[0].cpu().numpy()[0,0,:,:,:])

			for i in range(1, len(all_img)):
				all_img[i].append(processed[i-1][:,:,int(processed[i-1].shape[2]/2)])
				all_img_full[i].append(processed[i-1][:,:,:])

	return all_img_full

def get_correlation_features(model, dataset):
	features = {}
	correlations = {}

	df = pd.DataFrame(
		{'Transfer':[], 'Layer':[], 'Correlation':[]}
		)

	for c1 in dataset.label_list:
		features[c1] = compute_features(model, dataset, c1)

	for c1 in dataset.label_list: 
		for c2 in dataset.label_list:
			correlations[f'{c1}_{c2}'] = [[] for i in range(len(features[c1]))]

			for c in range(len(features[c1])):
				for img1, img2 in zip(features[c1][c], features[c2][c]):
					correlations[f'{c1}_{c2}'][c].append(get_correlation(img1, img2, nii=False))

				sub_df = pd.DataFrame(
					{'Transfer':[f'{c1}_{c2}'], 'Layer':[c], 'Correlation':[np.mean(correlations[f'{c1}_{c2}'][c])]}
					)

				df = pd.concat([df, sub_df])
	df.to_csv('./results/metrics/classifier-correlations.csv')
	return correlations