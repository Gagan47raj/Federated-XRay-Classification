import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]

sys.path.append(str(ROOT_DIR))

import torch

from src.models.baseline_cnn import (
    BaselineCNN
)

from src.evaluation.evaluate import (
    evaluate_model
)

from src.preprocessing.dataset_loader import (
    build_dataset_index
)

from src.preprocessing.splitter import (
    create_stratified_split
)

from src.preprocessing.dataloaders import (
    create_datasets,
    create_dataloaders
)

train_paths, train_labels = build_dataset_index(
    "../data/raw/train"
)

test_paths, test_labels = build_dataset_index(
    "../data/raw/test"
)

(
    X_train,
    X_val,
    y_train,
    y_val
) = create_stratified_split(
    train_paths,
    train_labels
)

(
    train_dataset,
    val_dataset,
    test_dataset
) = create_datasets(
    X_train,
    y_train,
    X_val,
    y_val,
    test_paths,
    test_labels
)

(
    train_loader,
    val_loader,
    test_loader
) = create_dataloaders(
    train_dataset,
    val_dataset,
    test_dataset
)

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

model = BaselineCNN()

model.load_state_dict(
    torch.load(
        "../src/models/best_baseline_cnn.pth",
        map_location=device
    )
)

model.to(device)

metrics, cm, report = evaluate_model(
    model,
    test_loader,
    device
)

print("\n=== TEST RESULTS ===\n")

for key, value in metrics.items():

    print(
        f"{key}: {value:.4f}"
    )

print("\n=== CONFUSION MATRIX ===\n")

print(cm)

print("\n=== CLASSIFICATION REPORT ===\n")

print(report)

import json

with open(
    "../results/metrics/baseline_results.json",
    "w"
) as f:

    json.dump(
        metrics,
        f,
        indent=4
    )

import matplotlib.pyplot as plt
plt.figure(figsize=(6, 6))

plt.imshow(cm)

plt.title(
    "Confusion Matrix"
)

plt.colorbar()

plt.xlabel(
    "Predicted"
)

plt.ylabel(
    "Actual"
)

plt.savefig(
    "../results/figures/confusion_matrix.png"
)

plt.show()