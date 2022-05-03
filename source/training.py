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
from source.utils.data_loading import get_data
from source.utils.data_loading import get_data
import matplotlib.pyplot as plt
from tqdm import tqdm


@torch.no_grad()
def evaluate(model, data_loader, criterion, device="cpu"):
    model.eval()
    outputs = [model.validation_step(batch, criterion, device) for batch in data_loader]
    return model.validation_epoch_end(outputs)


def fit(model, train_loader, val_loader, optimizer, criterion, epochs=10, device="cpu", is_logging=False, epoch_logging=5, trial_name=None):
    
    checkpoint_path = __generate_checkpoint_path(trial_name)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device)
    
    history = []
    for epoch in range(epochs):
        epoch_result = epoch_step(model, train_loader, val_loader, optimizer, criterion, epoch, device)
        history.append(epoch_result)

        if is_logging and (epoch + 1) % epoch_logging == 0:
          __log_result(history, model, optimizer, criterion, epoch, checkpoint_path)  

    return model, history


def epoch_step(model, train_loader, val_loader, optimizer, criterion, epoch, device="cpu"):

    # Training Phase 
    train_losses, train_accuracy = __train(model, train_loader, optimizer, criterion, device)

    # Validation phase
    val_result = __validate(model, val_loader, criterion, device)

    # Saving epoch's results
    epoch_result = __get_epochs_results(val_result, train_losses, train_accuracy)
    model.epoch_end(epoch, epoch_result)
    return epoch_result


def __train(model, train_loader, optimizer, criterion, device):
    model.train()
    train_losses = []
    train_accuracy = []
    for batch in tqdm(train_loader):
        loss, accu = model.training_step(batch, criterion, device)
        train_losses.append(loss)
        train_accuracy.append(accu)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    return train_losses, train_accuracy


def __validate(model, val_loader, criterion, device):
    return evaluate(model, val_loader, criterion, device)


def __get_epochs_results(val_result, train_losses, train_accuracy):
    result = val_result
    result['train_loss'] = torch.stack(train_losses).mean().item()
    result['train_accuracy'] = torch.stack(train_accuracy).mean().item()
    return result


def plot_result(history, path, epoch=None):
        Validation_accuracies = [x['Accuracy'] for x in history]
        Training_Accuracies = [x['train_accuracy'] for x in history]

        Validation_loss = [x['Loss'] for x in history]
        Training_loss = [x['train_loss'] for x in history]

        fig = plt.figure()
        ax0 = fig.add_subplot(121, title='loss')
        ax1 = fig.add_subplot(122, title='accuracy')

        box = ax1.get_position()
        box.x0 = box.x0 + 0.05
        box.x1 = box.x1 + 0.05
        ax1.set_position(box)
        
        ax1.plot(Training_Accuracies, '-rx', label='train')
        ax1.plot(Validation_accuracies, '-bx', label='val')

        ax0.plot(Training_loss, '-rx', label='train')
        ax0.plot(Validation_loss, '-bx', label='val')


        ax0.legend()
        ax1.legend()

        # Set common labels
        ax0.set_xlabel('epochs')
        ax0.set_ylabel('loss')

        ax1.set_xlabel('epochs')
        ax1.set_ylabel('accuracy')

        ax0.set_title('Loss vs no. of epochs')
        ax1.set_title('Acc. vs no. of epochs')

        ax0.xaxis.get_major_locator().set_params(integer=True)
        ax1.xaxis.get_major_locator().set_params(integer=True)

        if epoch != None and isinstance(epoch, int):
            file_name = f'loss_curve_{epoch}.jpg'
        else:
            file_name = 'loss_curve.jpg'

        fig.savefig(os.path.join(path, file_name))


def __generate_checkpoint_path(trial_name=None):
    base_path = ConfigManager().get_checkpoints_path()
    if trial_name is not None:
        dir_name = f'result_{trial_name}_{ConfigManager().get_now()}'
    else:
        dir_name = f'result_{ConfigManager().get_now()}'
    return os.path.join(base_path, dir_name)

def __log_result(history, model, optimizer, criterion, epoch, checkpoint_path):
    checkpoint_path = os.path.join(checkpoint_path, f'epoch_{epoch + 1}')
    if not os.path.isdir(checkpoint_path):
        os.makedirs(checkpoint_path)
    plot_result(history, checkpoint_path, epoch)
    cache_path = os.path.join(checkpoint_path, 'cache.pt')
    torch.save((model.state_dict(), optimizer.state_dict(), criterion.state_dict()), cache_path)
    with open(os.path.join(checkpoint_path, 'results.json'), 'w') as file:
        file.write(json.dumps(history))
