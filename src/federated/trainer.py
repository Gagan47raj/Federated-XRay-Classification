import sys
from pathlib import Path
import torch

ROOT = Path(r"C:\Users\gagan\Desktop\Federated Learning\federated-xray-classification")

sys.path.insert(0, str(ROOT))

import os

MODEL_DIR = Path(__file__).resolve().parents[1] / "models"
MODEL_DIR.mkdir(exist_ok=True)

from src.training.trainer import validate

def federated_training(
        server,
        clients,
        val_loader,
        criterion,
        device,
        rounds=10,
        local_epochs=1
):
    
        history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": []
        }

        best_val_acc = 0
        
        # Create models directory if it doesn't exist
        # os.makedirs("../models", exist_ok=True)

        for round_num in range(rounds):

            print(
                f"\n========== Round {round_num+1}/{rounds} =========="
            )

            server.distribute_weights(
                clients
            )

            round_loss = 0
            round_acc = 0

            for client in clients:

                loss, acc = client.train(
                    epochs=local_epochs
                )

                round_loss += loss
                round_acc += acc

                print(
                    f"{client.client_id} "
                    f"Loss={loss:.4f} "
                    f"Acc={acc:.4f}"
                )

            round_loss /= len(clients)
            round_acc /= len(clients)

            server.aggregate(
                clients
            )

            val_loss, val_acc = validate(
                server.global_model,
                val_loader,
                criterion,
                device
            )

            history["train_loss"].append(
                round_loss
            )

            history["train_acc"].append(
                round_acc
            )

            history["val_loss"].append(
                val_loss
            )

            history["val_acc"].append(
                val_acc
            )

            if val_acc > best_val_acc:

                best_val_acc = val_acc

                torch.save(
                    server.global_model.state_dict(),
                    MODEL_DIR / "federated_cbam_best.pth"
                )

                print(
                    f"Best model saved "
                    f"(Val Acc={val_acc:.4f})"
                )

            # Print metrics for every round (not just when best model is saved)
            print(
                f"Train Loss: {round_loss:.4f}"
            )

            print(
                f"Train Acc: {round_acc:.4f}"
            )

            print(
                f"Val Loss: {val_loss:.4f}"
            )

            print(
                f"Val Acc: {val_acc:.4f}"
            )

        # Return history after all rounds are complete
        return history