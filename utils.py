from tqdm import tqdm
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import copy
import pathlib
import torchvision
# import matplotlib as mpl
# mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# custom weights initialization
def weights_init(m):
    """
    Initialize

    Parameters
    ----------------------
    m: torch.nn.Module
        Module that means layer.
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:  # Conv系全てに対しての初期化
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)  # initialize for BN
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        # initialize Linear
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.normal_(0, 0.02)


def train_model(model, datasets, optimizer, criterion, num_epochs=30, batch_size=128,
                device=None, scheduler=None, out=None):
    """
    train gan(generator, discriminator) with standard gan algorithm

    Parameters
    -----------------
    models: torch.nn.Module
        pre-trained model

    datasets: torch.utils.data.Dataset
        dataset of image

    optimizer: torch.optim
        optimizer for model

    criterion: torch.nn.Module
        function that calculates loss 

    num_epochs: int
        number of epochs

    batch_size: int
        number of batch size

    device: torch.device

    out: pathlib.Path
        represent output directory

    Return
    -----------------------------
    model: torch.nn.Module
        best model
    """
    epochs = tqdm(range(num_epochs), desc="Epoch", unit='epoch')
    phases = ['train', 'val']
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    # construct dataloader
    dataloader = {phase: torch.utils.data.DataLoader(datasets[phase], batch_size=batch_size,
                                                     shuffle=(phase == 'train'), num_workers=2)
                  for phase in ['train', 'val']}
    dataset_sizes = {phase: len(datasets[phase]) for phase in ['train', 'val']}
    # train loop
    since = datetime.datetime.now()
    for epoch in epochs:
        for phase in phases:
            if phase == 'train':
                if scheduler is not None:
                    scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            train_loss = 0.0
            train_acc = 0.0
            # Iterate over data.
            iteration = tqdm(dataloader[phase],
                             desc="{} iteration".format(phase.capitalize()),
                             unit='iter')
            for inputs, labels in iteration:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()
                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    # returns loss is mean_wise
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                train_loss += loss.item() * inputs.size(0)
                train_acc += torch.sum(preds == labels.data)

            epoch_loss = train_loss / dataset_sizes[phase]
            epoch_acc = train_acc.double() / dataset_sizes[phase]
            tqdm.write('Epoch: {} Phase: {} Loss: {:.4f} Acc: {:.4f}'.format(
                epoch, phase.capitalize(), epoch_loss, epoch_acc))
            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        # save model by epoch
        torch.save(model.state_dict(), out / "model_{}epoch.pt".format(epoch))
        tqdm.write("")

    time_elapsed = datetime.datetime.now() - since
    tqdm.write('Training complete in {}'.format(time_elapsed))
    tqdm.write('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model
