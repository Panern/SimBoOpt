from typing import Dict
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader



'''
    This is for training and evaluation function for DL.
'''

def train(
        net: torch.nn.Module,
        train_loader: DataLoader,
        parameters: Dict[str, float],
        dtype: torch.dtype,
        device: torch.device,
        ) -> nn.Module :
    """
    Train CNN on provided data set.

    Args:
        net: initialized neural network
        train_loader: DataLoader containing training set
        parameters: dictionary containing parameters to be passed to the optimizer.
            - lr: default (0.001)
            - momentum: default (0.0)
            - weight_decay: default (0.0)
            - num_epochs: default (1)
        dtype: torch dtype
        device: torch device
    Returns:
        nn.Module: trained CNN.
    """
    # Initialize network
    net.to(dtype=dtype, device=device)  # pyre-ignore [28]
    net.train()
    # Define loss and optimizer
    criterion = nn.NLLLoss(reduction="sum")
    optimizer = optim.SGD(
            net.parameters(),
            lr=parameters.get("lr", 0.001),
            momentum=parameters.get("momentum", 0.0),
            weight_decay=parameters.get("weight_decay", 0.0),
            )
    scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(parameters.get("step_size", 30)),
            gamma=parameters.get("gamma", 1.0),  # default is no learning rate decay
            )

    num_epochs = parameters.get("num_epochs", 1)

    # Train Network
    for epochs in range(num_epochs) :
        for inputs, labels in train_loader :
            # move data to proper dtype and device
            inputs = inputs.to(dtype=dtype, device=device)
            labels = labels.to(device=device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
    return net

def evaluate(
        net: nn.Module, data_loader: DataLoader, dtype: torch.dtype, device: torch.device
        ) -> float :
    """
    Compute classification accuracy on provided dataset.

    Args:
        net: trained model
        data_loader: DataLoader containing the evaluation set
        dtype: torch dtype
        device: torch device
    Returns:
        float: classification accuracy
    """
    net.eval()
    correct = 0
    total = 0

    with torch.no_grad() :
        for inputs, labels in data_loader :
            # move data to proper dtype and device
            inputs = inputs.to(dtype=dtype, device=device)
            labels = labels.to(device=device)
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    ac = correct / total

    return ac




