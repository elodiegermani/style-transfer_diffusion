'''
This scripts implement functions to train a classifier.
'''

import torch
import torchvision as tv
import torchvision.transforms as transforms
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn import metrics

def train(model, train_dataset, distance, optimizer, device):
    '''
    Function to perform training of the CNN. 

    Parameters:
        - model, Classifier3D object: model trained.
        - train_dataset, ImageDataset object: training dataset.
        - distance: loss function used during training.
        - optimizer: optimizer used during training
        - device, torch.device: which device is beeing used.

    Returns:
        - mean_loss, float: mean loss on training dataset.
    '''
    model.train()

    with torch.set_grad_enabled(True):
        total_preds=[]
        total_labels=[]
        loss_total = 0
        
        for idx, data in enumerate(train_dataset):
            
            x_train = data[0]
            x_train = x_train.float().to(device)
            y_train = data[1]
            y_train = y_train.float().to(device)

            # clearing the Gradients of the model parameters
            optimizer.zero_grad()
            
            # prediction for training and validation set
            output_train = model(x_train)

            preds = torch.argmax(output_train, dim=1)
            labels = torch.argmax(y_train, 1)

            for i in preds:
                total_preds.append(i.item())
            for j in labels:
                total_labels.append(j.item())

            # computing the training and validation loss
            loss_train = distance(output_train, y_train)
            loss_total += loss_train.item()

            # computing the updated weights of all the model parameters
            loss_train.backward()
            optimizer.step()

        print(total_labels, total_preds)
        acc = metrics.accuracy_score(y_true = total_labels, y_pred = total_preds)
        print('Training accuracy:', acc)

        mean_loss = loss_total / len(train_dataset)

    return mean_loss

def validate(model, valid_dataset, distance, device):
    '''
    Function to perform validation during training of the autoencoder. 

    Parameters:
        - model, Classifier object: model trained.
        - valid_dataset, ImageDataset object: validation dataset.
        - distance: loss function used during training.
        - device, torch.device: which device is beeing used.

    Returns:
        - mean_loss, float: mean loss on validation dataset.
    '''
    total_preds=[]
    total_labels=[]
    model.eval()

    loss_total = 0

    for idx, data in enumerate(valid_dataset):
        x_val = data[0]
        x_val = x_val.float().to(device)
        y_val = data[1]
        y_val = y_val.float().to(device)
        
        # prediction for training and validation set
        output_val = model(x_val)

        preds = torch.argmax(output_val, dim=1)
        labels = torch.argmax(y_val, 1)
        
        for i in preds:
            total_preds.append(i.item())
        for j in labels:
            total_labels.append(j.item())

        # computing the training and validation loss
        loss_val = distance(output_val, y_val)
        loss_total += loss_val.item()

    mean_loss = loss_total / len(valid_dataset)

    print(total_labels, total_preds)

    acc = metrics.accuracy_score(y_true = total_labels, y_pred = total_preds)
    print('Validation accuracy:', acc)
    
    return mean_loss