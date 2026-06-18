import numpy as np


def create_client_partitions(
        image_paths,
        labels,
        num_clients=5,
        seed=42
):
    np.random.seed(seed)

    indices = np.arange(len(image_paths))
    np.random.shuffle(indices)

    client_indices = np.array_split(
        indices,
        num_clients
    )

    clients = {}
    
    for client_id, idx in enumerate(client_indices):

        client_paths = [
            image_paths[i]
            for i in idx
        ]

        client_labels = [
            labels[i]
            for i in idx
        ]

        clients[f"client_{client_id+1}"] = {
            "paths": client_paths,
            "labels": client_labels
        }

    return clients