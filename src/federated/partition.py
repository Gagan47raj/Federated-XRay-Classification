import numpy as np
from sklearn.model_selection import StratifiedKFold



def create_client_partitions(
        image_paths,
        labels,
        num_clients=5,
        seed=42
):

    image_paths = np.array(image_paths)
    labels = np.array(labels)

    skf = StratifiedKFold(
        n_splits=num_clients,
        shuffle=True,
        random_state=seed
    )

    clients = {}

    for i, (_, idx) in enumerate(
            skf.split(image_paths, labels)
    ):

        clients[f"client_{i+1}"] = {

            "paths": image_paths[idx].tolist(),

            "labels": labels[idx].tolist()

        }

    return clients