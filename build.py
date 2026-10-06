import os
import json
import torch

from PIL import Image
from torchvision import transforms

from augment import FaceAugmentor


# --------------------------------------------------
# Configuration
# --------------------------------------------------

AUGMENT_FACTOR = 6

torch.manual_seed(42)


# --------------------------------------------------
# Load split information
# --------------------------------------------------

with open("splits.json", "r") as f:
    splits = json.load(f)

class_names = splits["class_names"]

train_paths = splits["train"]["paths"]
train_labels = splits["train"]["labels"]

val_paths = splits["validation"]["paths"]
val_labels = splits["validation"]["labels"]


# --------------------------------------------------
# Create processed folder
# --------------------------------------------------

os.makedirs("processed", exist_ok=True)


# --------------------------------------------------
# Augmentor for training images
# --------------------------------------------------

augmentor = FaceAugmentor()


# --------------------------------------------------
# Validation transform
# No random augmentation
# --------------------------------------------------

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


# ==================================================
# BUILD TRAINING DATASET
# ==================================================

print("\nBuilding training dataset...")

train_images = []
final_train_labels = []
final_train_paths = []


for i, (image_path, label) in enumerate(
    zip(train_paths, train_labels),
    start=1
):

    print(
        f"Training image {i}/{len(train_paths)}: "
        f"{image_path}"
    )

    # Generate N random augmentations
    augmented_images = (
        augmentor.get_random_augmentations(
            image_path,
            n=AUGMENT_FACTOR
        )
    )

    for augmented_image in augmented_images:

        train_images.append(
            augmented_image
        )

        final_train_labels.append(
            label
        )

        # Store original source path
        final_train_paths.append(
            image_path
        )


# Stack into one tensor
train_images_tensor = torch.stack(
    train_images
)

train_labels_tensor = torch.tensor(
    final_train_labels,
    dtype=torch.long
)


# --------------------------------------------------
# Create training dictionary
# --------------------------------------------------

train_data = {

    "images": train_images_tensor,

    "labels": train_labels_tensor,

    "paths": final_train_paths,

    "class_names": class_names
}


# --------------------------------------------------
# Save train.pt
# --------------------------------------------------

train_output = "processed/train.pt"

torch.save(
    train_data,
    train_output
)


# ==================================================
# BUILD VALIDATION DATASET
# ==================================================

print("\nBuilding validation dataset...")

validation_images = []


for i, image_path in enumerate(
    val_paths,
    start=1
):

    print(
        f"Validation image {i}/{len(val_paths)}: "
        f"{image_path}"
    )

    image = Image.open(
        image_path
    ).convert("RGB")

    # Resize + tensor ONLY
    image = val_transform(
        image
    )

    validation_images.append(
        image
    )


# Stack validation images
val_images_tensor = torch.stack(
    validation_images
)

val_labels_tensor = torch.tensor(
    val_labels,
    dtype=torch.long
)


# --------------------------------------------------
# Create validation dictionary
# --------------------------------------------------

val_data = {

    "images": val_images_tensor,

    "labels": val_labels_tensor,

    "paths": val_paths,

    "class_names": class_names
}


# --------------------------------------------------
# Save val.pt
# --------------------------------------------------

val_output = "processed/val.pt"

torch.save(
    val_data,
    val_output
)


# ==================================================
# REPORT RESULTS
# ==================================================

print("\n====================================")
print("TRAIN.PT")
print("====================================")

print(
    "Images tensor shape:",
    train_images_tensor.shape
)

print(
    "Labels tensor shape:",
    train_labels_tensor.shape
)

print("\nSamples per class:")

for label, class_name in enumerate(
    class_names
):

    count = (
        train_labels_tensor == label
    ).sum().item()

    print(
        f"{class_name}: {count}"
    )


train_size_mb = (
    os.path.getsize(train_output)
    / (1024 * 1024)
)

print(
    f"\nFile size: "
    f"{train_size_mb:.2f} MB"
)


print("\n====================================")
print("VAL.PT")
print("====================================")

print(
    "Images tensor shape:",
    val_images_tensor.shape
)

print(
    "Labels tensor shape:",
    val_labels_tensor.shape
)

print("\nSamples per class:")

for label, class_name in enumerate(
    class_names
):

    count = (
        val_labels_tensor == label
    ).sum().item()

    print(
        f"{class_name}: {count}"
    )


val_size_mb = (
    os.path.getsize(val_output)
    / (1024 * 1024)
)

print(
    f"\nFile size: "
    f"{val_size_mb:.2f} MB"
)


print("\nDataset build complete.")

print(
    f"Training samples: "
    f"{len(train_labels_tensor)}"
)

print(
    f"Validation samples: "
    f"{len(val_labels_tensor)}"
)
