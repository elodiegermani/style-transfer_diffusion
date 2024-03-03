import numpy as np
import pandas as pd 
from math import log10, sqrt
import torch
import sys
import importlib
sys.path.insert(0, './feature_extractor')


def get_correlation(img1, img2):
    '''
    Compute the Pearson's correlation coefficient between original and reconstructed images.
    '''
    
    data1 = img1.get_fdata().copy()
    data2 = img2.get_fdata().copy()
    
    # Vectorise input data
    data1 = np.reshape(data1, -1)
    data2 = np.reshape(data2, -1)

    in_mask_indices = np.logical_not(
        np.logical_or(
            np.logical_or(np.isnan(data1), np.absolute(data1) == 0),
            np.logical_or(np.isnan(data2), np.absolute(data2) == 0)))

    data1 = data1[in_mask_indices] 
    data2 = data2[in_mask_indices]
    
    corr_coeff = np.corrcoef(data1, data2)[0][1]
    
    return corr_coeff


def PSNR(data1, data2):
	mse = np.mean((data1 - data2) ** 2)
	if(mse == 0):  # MSE is zero means no noise is present in the signal .
				  # Therefore PSNR have no importance.
		return 100
	max_pixel = 255.0
	psnr = 20 * log10(max_pixel / sqrt(mse))
	return psnr

def class_change(model_param, image):
	sys.path.insert(0, './feature_extractor')
	package = 'model'
	md = importlib.import_module(package)

	classifier = md.Classifier3D(
			n_class = 4
			)
	classifier.load_state_dict(
		torch.load(
			model_param, 
			map_location='cpu'
			)
		)


	classe = torch.max(classifier(image), 1)[1]

	return(classe)

def get_inception_score(p_yx, eps=1E-16):
	# calculate p(y)
	p_y = np.expand_dims(p_yx.mean(axis=0), 0)
	# kl divergence for each image
	kl_d = p_yx * (np.log(p_yx + eps) - np.log(p_y + eps))
	# sum over classes
	sum_kl_d = kl_d.sum(axis=1)
	# average over images
	avg_kl_d = np.mean(sum_kl_d)
	# undo the logs
	is_score = np.exp(avg_kl_d)
	
	return is_score