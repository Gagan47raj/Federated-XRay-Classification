
import torch

from src.training.metrics import (
    calculate_accuracy
)

def train_one_epoch(
    model,
    dataloader,
    criterion,
    optimizer,
    device
):

    model.train()

    running_loss = 0.0
    running_acc = 0.0

    for images, labels in dataloader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        acc = calculate_accuracy(
            outputs,
            labels
        )

        running_loss += loss.item()
        running_acc += acc

    epoch_loss = (
        running_loss /
        len(dataloader)
    )

    epoch_acc = (
        running_acc /
        len(dataloader)
    )

    return epoch_loss, epoch_acc

@torch.no_grad()
def validate(
    model,
    dataloader,
    criterion,
    device
):

    model.eval()

    running_loss = 0.0
    running_acc = 0.0

    for images, labels in dataloader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        acc = calculate_accuracy(
            outputs,
            labels
        )

        running_loss += loss.item()
        running_acc += acc

    epoch_loss = (
        running_loss /
        len(dataloader)
    )

    epoch_acc = (
        running_acc /
        len(dataloader)
    )

    return epoch_loss, epoch_acc