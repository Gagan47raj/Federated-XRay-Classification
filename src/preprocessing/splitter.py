# src/preprocessing/splitter.py

from sklearn.model_selection import train_test_split


def create_stratified_split(
    image_paths,
    labels,
    validation_size=0.20,
    random_state=42
):

    X_train, X_val, y_train, y_val = train_test_split(
        image_paths,
        labels,
        test_size=validation_size,
        random_state=random_state,
        stratify=labels
    )

    return (
        X_train,
        X_val,
        y_train,
        y_val
    )