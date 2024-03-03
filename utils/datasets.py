import pandas as pd
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset
from glob import glob
import pandas as pd
import os

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