import torch
import matplotlib.pyplot as plt
import random


# --------------------------------------------------
# Load datasets
# --------------------------------------------------

train_data = torch.load(
    "processed/train.pt",
    weights_only=False
)

val_data = torch.load(
    "processed/val.pt",
    weights_only=False
)


# --------------------------------------------------
# Extract data
# --------------------------------------------------

train_images = train_data["images"]
train_labels = train_data["labels"]

val_images = val_data["images"]
val_labels = val_data["labels"]

class_names = train_data["class_names"]


# --------------------------------------------------
# Print shapes
# --------------------------------------------------

print("\n====================================")
print("TRAIN DATA")
print("====================================")

print("Images shape:", train_images.shape)
print("Labels shape:", train_labels.shape)


print("\n====================================")
print("VALIDATION DATA")
print("====================================")

print("Images shape:", val_images.shape)
print("Labels shape:", val_labels.shape)


# --------------------------------------------------
# Class distributions using torch.unique
# --------------------------------------------------

train_unique, train_counts = torch.unique(
    train_labels,
    return_counts=True
)

val_unique, val_counts = torch.unique(
    val_labels,
    return_counts=True
)


print("\n====================================")
print("TRAIN CLASS DISTRIBUTION")
print("====================================")

for label, count in zip(
    train_unique.tolist(),
    train_counts.tolist()
):
    print(
        f"{label} -> "
        f"{class_names[label]}: "
        f"{count}"
    )


print("\n====================================")
print("VALIDATION CLASS DISTRIBUTION")
print("====================================")

for label, count in zip(
    val_unique.tolist(),
    val_counts.tolist()
):
    print(
        f"{label} -> "
        f"{class_names[label]}: "
        f"{count}"
    )


# --------------------------------------------------
# Convert counts into full lists
# --------------------------------------------------

train_class_counts = []

val_class_counts = []

for label in range(len(class_names)):

    train_count = (
        train_labels == label
    ).sum().item()

    val_count = (
        val_labels == label
    ).sum().item()

    train_class_counts.append(
        train_count
    )

    val_class_counts.append(
        val_count
    )


# --------------------------------------------------
# Bar chart: Train vs Validation
# --------------------------------------------------

x = range(len(class_names))

width = 0.35


plt.figure(figsize=(9, 6))

plt.bar(
    [i - width / 2 for i in x],
    train_class_counts,
    width,
    label="Train"
)

plt.bar(
    [i + width / 2 for i in x],
    val_class_counts,
    width,
    label="Validation"
)

plt.xticks(
    x,
    class_names
)

plt.xlabel("Class")
plt.ylabel("Number of Images")

plt.title(
    "Train vs Validation Class Distribution"
)

plt.legend()

plt.tight_layout()

plt.show()


# --------------------------------------------------
# Display 16 random training images
# --------------------------------------------------

number_of_images = min(
    16,
    len(train_images)
)

random.seed(42)

random_indices = random.sample(
    range(len(train_images)),
    number_of_images
)


plt.figure(
    figsize=(12, 12)
)


for i, index in enumerate(
    random_indices
):

    image = train_images[index]

    label = train_labels[index].item()

    # Convert:
    # C x H x W -> H x W x C
    image = image.permute(
        1,
        2,
        0
    )

    image = torch.clamp(
        image,
        0,
        1
    )


    plt.subplot(
        4,
        4,
        i + 1
    )

    plt.imshow(image)

    plt.title(
        class_names[label]
    )

    plt.axis("off")


plt.suptitle(
    "16 Random Training Images"
)

plt.tight_layout()

plt.show()