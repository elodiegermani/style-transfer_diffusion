import pandas as pd
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset
from glob import glob
import pandas as pd
import os
from glob import glob
from os.path import join as opj
from torch import default_generator, randperm, Generator
from torch._utils import _accumulate
import json
import random
import shutil

def create_dataset(data_dir, split=(800,100,100)):
    f_list = sorted(
        glob(
            os.path.join(data_dir, '*.nii')
        )
    )
    
    group = []
    pipeline = []
    contrast = []
    
    for f in f_list:
        group.append(
            f.split('/')[-1].split('_')[0].split('-')[1]
        )
        
        pipeline.append(
            f.split('/')[-1].split('_')[-2]
        )
        
        contrast.append(
            f.split('/')[-1].split('_')[1]
        )
        
    df_global = pd.DataFrame({
        'filepaths':f_list,
        'pipelines':pipeline,
        'groups':group,
        'contrast':contrast
    })
    
    train_groups = np.random.choice(
        np.unique(group), 
        size=800, replace=False
    )
    
    valid_test_groups = [
        i for i in np.unique(group) if i not in train_groups
    ]
    
    valid_groups = np.random.choice(
        valid_test_groups, 
        size=100, replace=False
    )
    
    test_groups = [
        i for i in valid_test_groups if i not in valid_groups
    ]
    
    assert(
        len([i for i in test_groups if i in train_groups])==0
    )
    assert(
        len([i for i in valid_groups if i in train_groups])==0
    )
    
    train_df = df_global.loc[
        df_global['groups'].isin(train_groups)
    ]
    test_df = df_global.loc[
        df_global['groups'].isin(test_groups)
    ]
    valid_df = df_global.loc[
        df_global['groups'].isin(valid_groups)
    ]
    
    train_df.to_csv('./data/train-dataset_rh.csv')
    test_df.to_csv('./data/test-dataset_rh.csv')
    valid_df.to_csv('./data/valid-dataset_rh.csv')

class ClassifDataset(Dataset):
    '''
    Create a Dataset object used to load training data and train a model using pytorch.

    Parameters:
        - data_dir, str: directory where images are stored
        - id_file, str: path to the text file containing ids of images of interest
        - label_file, str: path to the csv file containing labels of images of interest
        - label_column, str: name of the column to use as labels in label_file
        - label_list, list: list of unique labels sorted in alphabetical order

    Attributes:
        - data, list of str: list containing all images of the dataset selected
        - ids, list of int: list containing all ids of images of the selected dataset
        - labels, list of str: list containing all labels of each data
    '''
    def __init__(self, dataset_file, label_column):

        df = pd.read_csv(dataset_file)

        self.data = df['filepaths'].tolist()
        self.labels = df[label_column].tolist()
        self.groups = df['groups'].tolist()
        self.pipelines = df['pipelines'].tolist()
        self.contrast = df['contrast'].tolist()
        
        self.label_list = sorted(np.unique(self.labels))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        fname = self.data[idx]
        label = self.labels[idx]
        label_vect = [0 for i in range(len(self.label_list))]

        for i in range(len(self.label_list)):
            if label == self.label_list[i]:
                label_vect[i] = 1
        sample = nib.load(fname).get_fdata().copy().astype(float)
        sample = np.nan_to_num(sample)

        sample = torch.tensor(sample).view((1), *sample.shape)
        label_vect = torch.tensor(label_vect)
        
        return sample, label_vect

    def get_original_pipelines(self):
        return self.pipelines

    def get_original_labels(self):
        return self.labels

    def get_original_group(self):
        return self.groups
    
    def get_original_contrast(self):
        return self.contrast


class PairedImageDataset(Dataset):
    '''
    Create a Dataset object used to load training data and train the autoencoder (no labels needed).

    Parameters:
        - data_dir_A, str: directory where images of dataset A are stored
        - data_dir_B, str: directory where images of dataset B are stored
        - id_file, str: path to the text file containing ids of images of interest

    Attributes:
        - data_A, list of str: list containing all paths to images of the dataset A 
        - data_B, list of str: list containing all paths to images of the dataset B
        - ids, list of int: list containing all ids of images of the selected dataset
        - data, list of int: list containing all ids of images of the selected dataset
    '''
    def __init__(self, subset_A, subset_B, df_file, contrast_list):
        self.df = pd.read_csv(df_file)

        self.df_A = self.df[self.df['pipelines']==subset_A]
        self.df_B = self.df[self.df['pipelines']==subset_B]

        if contrast_list != 'All':
            self.df_A = self.df_A[self.df_A['contrast'].isin(contrast_list)]
            self.df_B = self.df_B[self.df_B['contrast'].isin(contrast_list)]
        
        self.data_A = self.df_A['filepaths'].tolist()
        self.data_B = self.df_B['filepaths'].tolist()

        self.ids = sorted(self.df_A['groups'].tolist())
        self.data = self.ids

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        '''
        Returns a pair of sample [data_A[idx], data_B[idx]] as Pytorch Tensors.
        '''
        fname_A = self.data_A[idx]
        sample_A = nib.load(fname_A).get_fdata().copy().astype(float)
        sample_A = np.nan_to_num(sample_A)
        sample_A = torch.tensor(sample_A).view((1), *sample_A.shape)

        fname_B = self.data_B[idx]
        sample_B = nib.load(fname_B).get_fdata().copy().astype(float)
        sample_B = np.nan_to_num(sample_B)
        sample_B = torch.tensor(sample_B).view((1), *sample_B.shape)
        
        return sample_A, sample_B

    def get_original_ids(self):
        return self.ids
    
    
class UnpairedImageDataset(Dataset):
    '''
    Create a Dataset object used to load training data and train the autoencoder (no labels needed).

    Parameters:
        - data_dir_A, str: directory where images of dataset A are stored
        - data_dir_B, str: directory where images of dataset B are stored
        - id_file, str: path to the text file containing ids of images of interest

    Attributes:
        - data_A, list of str: list containing all paths to images of the dataset A 
        - data_B, list of str: list containing all paths to images of the dataset B
        - ids, list of int: list containing all ids of images of the selected dataset
        - data, list of int: list containing all ids of images of the selected dataset
    '''
    def __init__(self, subset_A, subset_B, df_file, contrast_list):
        self.df = pd.read_csv(df_file)

        self.df_A = self.df[self.df['pipelines']==subset_A]
        self.df_B = self.df[self.df['pipelines']==subset_B]

        if contrast_list != 'All':
            self.df_A = self.df_A[self.df_A['contrast'].isin(contrast_list)]
            self.df_B = self.df_B[self.df_B['contrast'].isin(contrast_list)]
        
        self.data_A = self.df_A['filepaths'].tolist()
        self.data_B = self.df_B['filepaths'].tolist()

        self.ids = sorted(self.df_A['groups'].tolist())
        self.data = self.ids

        random.shuffle(self.data_A)
        random.shuffle(self.data_B)

        assert(len(self.data_A)==len(self.data_B))
        
        self.data = [(self.data_A[i], self.data_B[i]) for i in range(len(self.data_A))]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        '''
        Returns a pair of sample [data_A[idx], data_B[idx]] as Pytorch Tensors.
        '''
        fname_A = self.data_A[idx]
        sample_A = nib.load(fname_A).get_fdata().copy().astype(float)
        sample_A = np.nan_to_num(sample_A)
        sample_A = sample_A[0:48,0:56,0:48]
        sample_A = torch.tensor(sample_A).view((1), *sample_A.shape)

        fname_B = self.data_B[idx]
        sample_B = nib.load(fname_B).get_fdata().copy().astype(float)
        sample_B = np.nan_to_num(sample_B)
        sample_B = sample_B[0:48,0:56,0:48]
        sample_B = torch.tensor(sample_B).view((1), *sample_B.shape)
        
        return sample_A, sample_B

    def get_original_ids(self):
        return self.data_A, self.data_B

class ImageDataset(Dataset):
    '''
    Create a Dataset object used to load training data and train a model using pytorch.

    Parameters:
        - file_list, list of str: list of all images

    Attributes:
        - data, list of str: list containing all images of the dataset selected
    '''
    def __init__(self, file_list):

        self.data = file_list
        

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        fname = self.data[idx]

        sample = nib.load(fname).get_fdata().copy().astype(float)
        sample = np.nan_to_num(sample)
        sample = torch.tensor(sample).view((1), *sample.shape)
        
        return sample