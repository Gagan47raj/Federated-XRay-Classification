import copy

import torch
import torch.nn as nn
import torch.optim as optim

from src.models.cbam_cnn import CBAMCNN
from src.training.trainer import train_one_epoch

class FederatedClient:

    def __init__(
            self,
            client_id,
            train_loader,
            device,
            class_weights=None,
            lr=1e-3
    ):

        self.client_id = client_id

        self.device = device

        self.train_loader = train_loader

        self.model = CBAMCNN()

        self.model.to(device)

        self.criterion = nn.CrossEntropyLoss(
            weight=class_weights
        )

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=lr
        )

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=2
        )

    def train(
            self,
            epochs=1
    ):

        losses = []
        accuracies = []

        for _ in range(epochs):

            loss, acc = train_one_epoch(
                self.model,
                self.train_loader,
                self.criterion,
                self.optimizer,
                self.device
            )

            self.scheduler.step(loss)

            losses.append(loss)
            accuracies.append(acc)

        return (
            sum(losses)/len(losses),
            sum(accuracies)/len(accuracies)
        )

    def get_weights(self):

        return copy.deepcopy(
            self.model.state_dict()
        )

    def set_weights(
            self,
            weights
    ):

        self.model.load_state_dict(
            weights
        )