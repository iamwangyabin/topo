import os, sys
sys.path.append(os.getcwd())

import torch
from torch import nn
from torch import autograd


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        # input 1,180,60
        con1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1, stride=1),
            nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
        )
        # output 32,90,30
        con2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, padding=1, stride=2),
            nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
        )
        # 64,22,7
        con3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, padding=1, stride=2),
            nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
        )
        # 64,5,1

        con4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=1, padding=1, stride=1),
            nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
        )
        # 128,3,1

        con5 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=1, padding=1, stride=1),
            nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
        )
        # 256,2,1

        dens1 = nn.Sequential(
            nn.Linear(in_features=64*5,out_features=10),
            nn.Linear(in_features=10, out_features=1),

        )
        dens2 = nn.Sequential(
            nn.Linear(in_features=128*3,out_features=10),
            nn.Linear(in_features=10,out_features=1),
        )
        dens3 = nn.Sequential(
            nn.Linear(in_features=256*2,out_features=50),
            nn.Linear(in_features=50,out_features=5),
            nn.Linear(in_features=5,out_features=1),
        )
        self.con1 = con1
        self.con2 = con2
        self.con3 = con3
        self.dense1=dens1
        self.con4 = con4
        self.dense2=dens2
        self.con5 = con5
        self.dense3=dens3

    def forward(self, input):
        output = self.con1(input)
        output = self.con2(output)
        output = self.con3(output)
        x = output.view(-1, self.num_flat_features(output))
        volfrac = self.dense1(x)
        output = self.con4(output)
        x = output.view(-1, self.num_flat_features(output))
        penal = self.dense2(x)
        output = self.con5(output)
        x = output.view(-1, self.num_flat_features(output))
        rmin = self.dense3(x)
        return volfrac,penal,rmin

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


#net = Generator()

#optimizerG = optim.Adam(netG.parameters(), lr=1e-4, betas=(0.5, 0.9))

# For generating samples
def generate_image(netG):
    fixed_noise_128 = torch.randn(120, 120)
    noisev = autograd.Variable(fixed_noise_128, volatile=True)
    samples = netG(noisev)
    samples = samples.view(-1, 1, 32, 32)
    samples = samples.mul(0.5).add(0.5)
    samples = samples.cpu().data.numpy()
    return samples

