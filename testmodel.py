import os, sys
sys.path.append(os.getcwd())

import time
import numpy as np

import torch
from torch import nn
from torch import autograd
from torch import optim
import torchvision
DIM = 128 # This overfits substantially; you're probably better off with 64


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        preprocess = nn.Sequential(
            nn.Linear(128, 4 * 4 * 4 * DIM),
            nn.BatchNorm1d(4 * 4 * 4 * DIM),
            nn.ReLU(True),
        )
        self.preprocess = preprocess

    def forward(self, input):
        output = self.preprocess(input)
        output = output.view(-1, 4 * DIM, 4, 4)
        return output

k=Generator()
fixed_noise_128 = torch.randn(128, 128)
k(fixed_noise_128)

def generate_image(netG):
    fixed_noise_128 = torch.randn(128, 128)
    if use_cuda:
        fixed_noise_128 = fixed_noise_128.cuda(gpu)
    noisev = autograd.Variable(fixed_noise_128, volatile=True).unsqueeze(0).unsqueeze(0)
    samples = netG(noisev)
    samples = samples.view(-1, 3, 32, 32)
    samples = samples.mul(0.5).add(0.5)
    samples = samples.cpu().data.numpy()
    return samples
