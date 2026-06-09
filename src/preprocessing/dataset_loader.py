from pathlib import Path
from PIL import Image

from torch.utils.data import Dataset

class ChestXRayDataset(Dataset):

    def __init__(self, image_paths, labels, transform=None):

        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):

        return len(self.image_paths)

    def __getitem__(self, idx):

        image = Image.open(
            self.image_paths[idx]
        ).convert("L")

        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label
    
def get_label_from_folder(folder_name):

    label_map = {
        "NORMAL": 0,
        "PNEUMONIA": 1
    }

    return label_map[folder_name]

def build_dataset_index(dataset_dir):

    image_paths = []
    labels = []

    dataset_dir = Path(dataset_dir)

    for class_folder in [
        "NORMAL",
        "PNEUMONIA"
    ]:

        class_path = dataset_dir / class_folder

        for image_path in class_path.glob("*"):

            image_paths.append(
                str(image_path)
            )

            labels.append(
                get_label_from_folder(
                    class_folder
                )
            )

    return image_paths, labels

CLASS_NAMES = {
    0: "NORMAL",
    1: "PNEUMONIA"
}

