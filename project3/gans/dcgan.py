from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.utils as vutils
import matplotlib.pyplot as plt

import sys
import numpy as np
import os


class Generator (nn.Module):
    def __init__(self, ngf, nz, nc):
        super(Generator, self).__init__()
        self.ngf = ngf  # depth of feature maps carried through the generator
        self.nz = nz  # length of latent vector
        self.nc = nc  # number of color channels in the input images

        # net
        # conv layers
        self.cnn1 = nn.ConvTranspose2d(
            self.nz, self.ngf * 8, kernel_size=4, stride=1, padding=0, bias=False)
        self.cnn2 = nn.ConvTranspose2d(
            self.ngf * 8, self.ngf * 4, kernel_size=4, stride=2, padding=1, bias=False)
        self.cnn3 = nn.ConvTranspose2d(
            self.ngf * 4, self.ngf * 2, kernel_size=4, stride=2, padding=1, bias=False)
        self.cnn4 = nn.ConvTranspose2d(
            self.ngf * 2, self.ngf, kernel_size=4, stride=2, padding=1, bias=False)
        self.cnn5 = nn.ConvTranspose2d(
            self.ngf, self.nc, kernel_size=4, stride=2, padding=1, bias=False)

        # batch normalizatio
        self.bn1 = nn.BatchNorm2d(self.ngf * 8)
        self.bn2 = nn.BatchNorm2d(self.ngf * 4)
        self.bn3 = nn.BatchNorm2d(self.ngf * 2)
        self.bn4 = nn.BatchNorm2d(self.ngf)

    def forward(self, x):
        x = F.relu(self.bn1(self.cnn1(x)))
        x = F.relu(self.bn2(self.cnn2(x)))
        x = F.relu(self.bn3(self.cnn3(x)))
        x = F.relu(self.bn4(self.cnn4(x)))
        x = F.tanh(self.cnn5(x))
        return x

    def generate_image(self, batch_size, nz, device):
        noise = torch.randn(batch_size, nz, 1, 1, device=device)
        image = self(noise).detach().cpu()
        return image

    def display_generated_image(self, batch_size=1, nz=100, device='cpu'):
        image = self.generate_image(batch_size, nz, device)
        gen_img = vutils.make_grid(image, padding=2, normalize=True)
        fig = plt.figure(figsize=(8, 8))
        plt.axis("off")
        plt.imshow(np.transpose(gen_img, (1, 2, 0)))
        plt.show()
    def save(self,folder):
        torch.save(self.state_dict(), os.path.join(folder,'generator'))


class Discriminator(nn.Module):
    def __init__(self, ndf, nc):
        super(Discriminator, self).__init__()
        self.ndf = ndf  # depth of feature maps propagated through the discriminator
        self.nc = nc  # number of color channels in the input images

        # net
        # conv layers
        self.cnn1 = nn.Conv2d(self.nc, self.ndf, kernel_size=4,
                              stride=2, padding=1, bias=False)
        self.cnn2 = nn.Conv2d(self.ndf, self.ndf * 2,
                              kernel_size=4, stride=2, padding=1, bias=False)
        self.cnn3 = nn.Conv2d(self.ndf * 2, self.ndf * 4,
                              kernel_size=4, stride=2, padding=1, bias=False)
        self.cnn4 = nn.Conv2d(self.ndf * 4, self.ndf * 8,
                              kernel_size=4, stride=2, padding=1, bias=False)
        self.cnn5 = nn.Conv2d(self.ndf * 8, 1, kernel_size=4,
                              stride=1, padding=0, bias=False)

        # batch norms
        self.bn1 = nn.BatchNorm2d(self.ndf * 2)
        self.bn2 = nn.BatchNorm2d(self.ndf * 4)
        self.bn3 = nn.BatchNorm2d(self.ndf * 8)

        # leaky relu
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x = self.lrelu(self.cnn1(x))
        x = self.lrelu(self.bn1(self.cnn2(x)))
        x = self.lrelu(self.bn2(self.cnn3(x)))
        x = self.lrelu(self.bn3(self.cnn4(x)))
        x = F.sigmoid(self.cnn5(x))
        return x
    def save(self,folder):
        torch.save(self.state_dict(), os.path.join(folder,'discriminator'))
