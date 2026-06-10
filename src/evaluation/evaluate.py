
import json
import torch
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

@torch.no_grad()
def evaluate_model(
    model,
    dataloader,
    device
):

    model.eval()

    all_preds = []
    all_labels = []

    all_probs = []

    for images, labels in dataloader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        probs = torch.softmax(
            outputs,
            dim=1
        )

        preds = torch.argmax(
            outputs,
            dim=1
        )

        all_preds.extend(
            preds.cpu().numpy()
        )

        all_labels.extend(
            labels.cpu().numpy()
        )

        all_probs.extend(
            probs[:, 1].cpu().numpy()
        )

    accuracy = accuracy_score(
        all_labels,
        all_preds
    )

    precision = precision_score(
        all_labels,
        all_preds
    )

    recall = recall_score(
        all_labels,
        all_preds
    )

    f1 = f1_score(
        all_labels,
        all_preds
    )

    roc_auc = roc_auc_score(
        all_labels,
        all_probs
    )

    cm = confusion_matrix(
        all_labels,
        all_preds
    )

    report = classification_report(
        all_labels,
        all_preds
    )

    metrics = {

        "accuracy": float(accuracy),

        "precision": float(precision),

        "recall": float(recall),

        "f1_score": float(f1),

        "roc_auc": float(roc_auc)
    }

    return (
        metrics,
        cm,
        report
    )