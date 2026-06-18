import copy
import torch

from src.models.cbam_cnn import CBAMCNN
from src.federated.fedAvg import federated_average


class FederatedServer:

    def __init__(self, device):

        self.device = device

        self.global_model = CBAMCNN()

        self.global_model.to(device)

    def get_global_weights(self):

        return copy.deepcopy(
            self.global_model.state_dict()
        )
    
    def set_global_weights(
        self,
        weights
    ):

        self.global_model.load_state_dict(
            weights
        )

    def distribute_weights(
        self,
        clients
    ):

        global_weights = self.get_global_weights()

        for client in clients:

            client.set_weights(
                global_weights
            )

    def collect_weights(
        self,
        clients
    ):

        client_weights = []

        for client in clients:

            client_weights.append(
                client.get_weights()
            )

        return client_weights
    
    def aggregate(
            self,
            clients
    ):

        client_weights = self.collect_weights(
            clients
        )

        global_weights = federated_average(
            client_weights
        )

        self.set_global_weights(
            global_weights
        )