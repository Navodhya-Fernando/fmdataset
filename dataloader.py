import os
import torch

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# --------------------------------------------------
# Custom Dataset
# --------------------------------------------------

class FaceDataset(Dataset):

    def __init__(self, pt_file, transform=None):

        data = torch.load(
            pt_file,
            weights_only=False
        )

        self.images = data["images"]
        self.labels = data["labels"]
        self.class_names = data["class_names"]

        self.transform = transform


    def __len__(self):

        return len(self.images)


    def __getitem__(self, index):

        image = self.images[index]
        label = self.labels[index]

        if self.transform:
            image = self.transform(image)

        return image, label


# --------------------------------------------------
# Create DataLoaders
# --------------------------------------------------

def create_dataloaders(
    dataset_root=".",
    batch_size=8
):

    train_path = os.path.join(
        dataset_root,
        "processed",
        "train.pt"
    )

    val_path = os.path.join(
        dataset_root,
        "processed",
        "val.pt"
    )


    # ImageNet normalization statistics
    normalize = transforms.Normalize(
        mean=[
            0.485,
            0.456,
            0.406
        ],
        std=[
            0.229,
            0.224,
            0.225
        ]
    )


    train_dataset = FaceDataset(
        train_path,
        transform=normalize
    )

    val_dataset = FaceDataset(
        val_path,
        transform=normalize
    )


    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )


    class_names = train_dataset.class_names

    return (
        train_loader,
        val_loader,
        class_names
    )


# --------------------------------------------------
# Test
# --------------------------------------------------

if __name__ == "__main__":

    train_loader, val_loader, class_names = (
        create_dataloaders(
            ".",
            batch_size=8
        )
    )


    images, labels = next(
        iter(train_loader)
    )


    print(
        "Batch shape:",
        images.shape
    )

    print(
        "Labels:",
        labels
    )

    print(
        "Label names:",
        [
            class_names[label.item()]
            for label in labels
        ]
    )


    print(
        "\nNumber of training batches:",
        len(train_loader)
    )

    print(
        "Number of validation batches:",
        len(val_loader)
    )