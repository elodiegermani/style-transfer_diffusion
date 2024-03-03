'''
This scripts implement a conditional DDPM.

Diffusion model is based on DDPM,
https://arxiv.org/abs/2006.11239

The conditioning idea is taken from 'Classifier-Free Diffusion Guidance',
https://arxiv.org/abs/2207.12598
'''

from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
from models.unet import ContextUnet

def ddpm_schedules(beta1, beta2, T):
    """
    Returns pre-computed schedules for DDPM sampling, training process.
    """
    assert beta1 < beta2 < 1.0, "beta1 and beta2 must be in (0, 1)"

    beta_t = (beta2 - beta1) * torch.arange(0, T + 1, dtype=torch.float32) / T + beta1
    sqrt_beta_t = torch.sqrt(beta_t)
    alpha_t = 1 - beta_t
    log_alpha_t = torch.log(alpha_t)
    alphabar_t = torch.cumsum(log_alpha_t, dim=0).exp()

    sqrtab = torch.sqrt(alphabar_t)
    oneover_sqrta = 1 / torch.sqrt(alpha_t)

    sqrtmab = torch.sqrt(1 - alphabar_t)
    mab_over_sqrtmab_inv = (1 - alpha_t) / sqrtmab

    return {
        "alpha_t": alpha_t,  # \alpha_t
        "oneover_sqrta": oneover_sqrta,  # 1/\sqrt{\alpha_t}
        "sqrt_beta_t": sqrt_beta_t,  # \sqrt{\beta_t}
        "alphabar_t": alphabar_t,  # \bar{\alpha_t}
        "sqrtab": sqrtab,  # \sqrt{\bar{\alpha_t}}
        "sqrtmab": sqrtmab,  # \sqrt{1-\bar{\alpha_t}}
        "mab_over_sqrtmab": mab_over_sqrtmab_inv,  # (1-\alpha_t)/\sqrt{1-\bar{\alpha_t}}
    }


class DDPM(nn.Module):
    def __init__(self, config):
        super(DDPM, self).__init__()

        self.nn_model = ContextUnet(in_channels=1, n_feat=config.n_feat, n_classes=config.n_classes)
        self.betas = config.beta 
        self.n_T = config.n_T
        self.drop_prob = config.drop_prob
        self.n_classes = config.n_classes

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.nn_model.to(self.device)

        # register_buffer allows accessing dictionary produced by ddpm_schedules
        # e.g. can access self.sqrtab later
        for k, v in ddpm_schedules(self.betas[0], self.betas[1], self.n_T).items():
            self.register_buffer(k, v)

        self.loss_mse = nn.MSELoss()

    def forward(self, x, c):
        """
        this method is used in training, so samples t and noise randomly
        """
        # for sampling noise and real 
        _ts = torch.randint(1, self.n_T+1, (x.shape[0],)).to(self.device)  # t ~ Uniform(0, n_T)
        noise = torch.randn_like(x)  # eps ~ N(0, 1)

        x_t = (
            self.sqrtab.to(self.device)[_ts, None, None, None, None] * x
            + self.sqrtmab.to(self.device)[_ts, None, None, None, None] * noise
        )  # This is the x_t, which is sqrt(alphabar) x_0 + sqrt(1-alphabar) * eps
        # We should predict the "error term" from this x_t. Loss is what we return.

        # dropout context with some probability
        ## context_mask = tensor of size n. image with 0/1 if drop context.
        context_mask = torch.bernoulli(torch.zeros_like(torch.argmax(c, dim=1))+self.drop_prob).to(self.device)
        
        # return MSE between added noise, and our predicted noise
        return self.loss_mse(noise, self.nn_model(x_t, c, _ts / self.n_T, context_mask))

    def sample(self, n_sample, size, guide_w = 0.0):
        # we follow the guidance sampling scheme described in 'Classifier-Free Diffusion Guidance'
        # to make the fwd passes efficient, we concat two versions of the dataset,
        # one with context_mask=0 and the other context_mask=1
        # we then mix the outputs with the guidance scale, w
        # where w>0 means more guidance

        x_i = torch.randn(n_sample, *size).to(self.device)  # x_T ~ N(0, 1), sample initial noise
        c_i = torch.arange(0,self.n_classes).to(self.device) # context for us just cycles throught the labels
        c_i = c_i.repeat(int(n_sample/c_i.shape[0]))

        #c_i = nn.functional.one_hot(c_i, num_classes=24).to(self.device)

        # don't drop context at test time
        context_mask = torch.zeros_like(c_i).to(self.device)

        # double the batch
        c_i = c_i.repeat(2)
        context_mask = context_mask.repeat(2)
        context_mask[n_sample:] = 1.

        x_i_store = [] # keep track of generated steps in case want to plot something 

        for i in range(self.n_T, 0, -1):
            print(f'sampling timestep {i}',end='\r')
            t_is = torch.tensor([i / self.n_T]).to(self.device)
            t_is = t_is.repeat(n_sample,1,1,1,1)

            # double batch
            x_i = x_i.repeat(2,1,1,1,1)
            t_is = t_is.repeat(2,1,1,1,1)

            z = torch.randn(n_sample, *size).to(self.device) if i > 1 else 0

            # split predictions and compute weighting  
            ci_vect = nn.functional.one_hot(c_i, num_classes=self.n_classes).to(self.device)    

            eps = self.nn_model(x_i.float(), ci_vect.float(), t_is.float(), context_mask.float())
            eps1 = eps[:n_sample] # first part (context_mask = 0)
            eps2 = eps[n_sample:] # second part (context_mask = 1)
            eps = (1+guide_w)*eps1 - guide_w*eps2 # mix output: context mask off and context mask on
            x_i = x_i[:n_sample] # Keep half of the samples 
            x_i = (
                self.oneover_sqrta[i] * (x_i - eps * self.mab_over_sqrtmab[i])
                + self.sqrt_beta_t[i] * z
            ) 
            
            if i%20==0 or i==self.n_T or i<8:
                x_i_store.append(x_i.detach().cpu().numpy())
        
        x_i_store = np.array(x_i_store)
        return x_i, x_i_store

    def transfer(self, source, c_t, guide_w = 0.0):
        # we follow the guidance sampling scheme described in 'Classifier-Free Diffusion Guidance'
        # to make the fwd passes efficient, we concat two versions of the dataset,
        # one with context_mask=0 and the other context_mask=1
        # we then mix the outputs with the guidance scale, w
        # where w>0 means more guidance

        x_i = source.to(self.device)  # 
        noise = torch.randn_like(x_i)  # eps ~ N(0, 1)
        x_t = (
            self.sqrtab.to(self.device)[self.n_T] * x_i
            + self.sqrtmab.to(self.device)[self.n_T] * noise
        )

        c_t = torch.argmax(c_t, dim=1).to(self.device) # Target class

        # don't drop context at test time
        context_mask = torch.zeros_like(c_t).to(self.device)

        # double the batch
        c_t = c_t.repeat(2)
        context_mask = context_mask.repeat(2)
        context_mask[1:] = 1.

        for i in range(self.n_T, 0, -1):

            print(f'sampling timestep {i}',end='\r')
            t_is = torch.tensor([i / self.n_T]).to(self.device)
            t_is = t_is.repeat(1,1,1,1,1)

            z = torch.randn(*x_t.shape).to(self.device) if i > 1 else 0

            # double batch
            x_t = x_t.repeat(2,1,1,1,1)
            t_is = t_is.repeat(2,1,1,1,1)

            # split predictions and compute weighting  
            ct_vect = nn.functional.one_hot(c_t, num_classes=self.n_classes).to(self.device)  

            eps = self.nn_model(x_t.float(), ct_vect.float(), t_is.float(), context_mask.float())
            eps1 = eps[:1] # first part (context_mask = 0)
            eps2 = eps[1:] # second part (context_mask = 1)
            eps = (1+guide_w)*eps1 - guide_w*eps2 # mix output: context mask off and context mask on
            x_t = x_t[0:1] # Keep half of the samples 

            x_t = (
                self.oneover_sqrta[i] * (x_t - eps * self.mab_over_sqrtmab[i])
                + self.sqrt_beta_t[i] * z
            ) 
        
        return x_t