from cProfile import label
import json
import os
from random import sample

from numpy import integer
from source.utils.config_manager import ConfigManager
import torch
import torch.nn as nn
import ray
from ray import tune
import torchvision.utils as vutils
from source.utils.data_loading import get_data
from source.utils.data_loading import get_data
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

real_label = 1.
fake_label = 0.


def fit(modelG, modelD, train_loader, optimizerG, optimizerD, criterion, fixed_noise, epochs=10, device="cpu", is_logging=False, epoch_logging=5, trial_name=None, checkpoint_path=None):

    # if checkpoint_path is None:
    #     checkpoint_path = generate_checkpoint_path(trial_name)
    img_list = []
    if torch.cuda.device_count() > 1:
        modelG = nn.DataParallel(modelG)
        modelD = nn.DataParallel(modelD)
    modelG.to(device)
    modelD.to(device)

    history = []
    for epoch in range(epochs):
        epoch_result = epoch_step(
            modelG, modelD, train_loader, optimizerG, optimizerD, criterion, epoch, device)
        history.append(epoch_result)

    with torch.no_grad():
        fake = modelG(fixed_noise).detach().cpu()
    img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
    return modelG, modelD, history, img_list


def epoch_step(modelG, modelD, train_loader, optimizerG, optimizerD, criterion, epoch, device="cpu"):

    # Training Phase
    G_losses, D_losses = _train(
        modelG, modelD, train_loader, optimizerG, optimizerD, criterion, device)

    # Saving epoch's results
    epoch_result = __get_epochs_results(G_losses, D_losses)
    _epoch_end(epoch, epoch_result)
    return epoch_result


def __get_epochs_results(G_losses, D_losses):
    result = {}
    result['G_losses'] = G_losses
    result['D_losses'] = D_losses
    return result


def _epoch_end(epoch, result):
    print(result)
    print("Epoch :", epoch + 1)



def _train(modelG, modelD, train_loader, optimizerG, optimizerD, criterion, device):
    G_losses = []
    D_losses = []
    for i, data in tqdm(enumerate(train_loader, 0), total=5000):
        if i > 10000:
            break
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        # Train with all-real batch
        modelD.zero_grad()
        # Format batch
        real_cpu = data[0].to(device)
        b_size = real_cpu.size(0)
        label = torch.full((b_size,), real_label ,
                           dtype=torch.float, device=device) + torch.FloatTensor((b_size,)).uniform_(-0.2,0.2)
        # Forward pass real batch through D
        output = modelD(real_cpu).view(-1)
        # Calculate loss on all-real batch
        errD_real = criterion(output, label)
        # Calculate gradients for D in backward pass
        errD_real.backward()
        D_x = output.mean().item()

        # Train with all-fake batch
        # Generate batch of latent vectors
        noise = torch.randn(b_size, modelG.nz, 1, 1, device=device)
        # Generate fake image batch with G
        fake = modelG(noise)
        label.fill_(fake_label)
        # Classify all fake batch with D
        output = modelD(fake.detach()).view(-1)
        # Calculate D's loss on the all-fake batch
        errD_fake = criterion(output, label)
        # Calculate the gradients for this batch, accumulated (summed) with previous gradients
        errD_fake.backward()
        #D_G_z1 = output.mean().item()
        # Compute error of D as sum over the fake and the real batches
        errD = errD_real + errD_fake
        # Update D
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        modelG.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost
        # Since we just updated D, perform another forward pass of all-fake batch through D
        output = modelD(fake).view(-1)
        # Calculate G's loss based on this output
        errG = criterion(output, label)
        # Calculate gradients for G
        errG.backward()
        #D_G_z2 = output.mean().item()
        # Update G
        optimizerG.step()

        # Save Losses for plotting later
        G_losses.append(errG.item())
        D_losses.append(errD.item())
    return G_losses, D_losses
