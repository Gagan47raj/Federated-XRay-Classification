from torch.utils.data import DataLoader

from src.preprocessing.dataset_loader import (
    ChestXRayDataset
)

from src.preprocessing.transforms import (
    train_transform,
    val_transform,
    test_transform
)

def create_datasets(
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test
):

    train_dataset = ChestXRayDataset(
        X_train,
        y_train,
        transform=train_transform
    )

    val_dataset = ChestXRayDataset(
        X_val,
        y_val,
        transform=val_transform
    )

    test_dataset = ChestXRayDataset(
        X_test,
        y_test,
        transform=test_transform
    )

    return (
        train_dataset,
        val_dataset,
        test_dataset
    )

def create_dataloaders(
    train_dataset,
    val_dataset,
    test_dataset,
    batch_size=32,
    num_workers=2
):

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return (
        train_loader,
        val_loader,
        test_loader
    )

