import os
import json
import random
from collections import Counter

from sklearn.model_selection import train_test_split


# --------------------------------------------------
# Load class names
# --------------------------------------------------

with open("config.json", "r") as f:
    config = json.load(f)

class_names = config["class_names"]


# --------------------------------------------------
# Collect image paths and labels
# --------------------------------------------------

image_paths = []
labels = []

valid_extensions = (".jpg", ".jpeg", ".png")

for label, class_name in enumerate(class_names):

    class_folder = os.path.join("raw", class_name)

    if not os.path.exists(class_folder):
        print(f"Warning: Folder not found: {class_folder}")
        continue

    files = sorted(os.listdir(class_folder))

    for filename in files:

        if filename.lower().endswith(valid_extensions):

            image_path = os.path.join(
                class_folder,
                filename
            )

            image_paths.append(image_path)
            labels.append(label)


# --------------------------------------------------
# Check collected data
# --------------------------------------------------

print("\nTotal images:", len(image_paths))

print("\nImages per class:")

for label, class_name in enumerate(class_names):

    count = labels.count(label)

    print(
        f"{label} -> {class_name}: {count}"
    )


# --------------------------------------------------
# Shuffle using fixed seed
# --------------------------------------------------

random.seed(42)

combined = list(
    zip(image_paths, labels)
)

random.shuffle(combined)

image_paths, labels = zip(*combined)

image_paths = list(image_paths)
labels = list(labels)


# --------------------------------------------------
# 80 / 20 stratified split
# --------------------------------------------------

train_paths, val_paths, train_labels, val_labels = (
    train_test_split(
        image_paths,
        labels,
        test_size=0.20,
        random_state=42,
        stratify=labels
    )
)


# --------------------------------------------------
# Save split information
# --------------------------------------------------

splits = {

    "train": {
        "paths": train_paths,
        "labels": train_labels
    },

    "validation": {
        "paths": val_paths,
        "labels": val_labels
    },

    "class_names": class_names
}


with open("splits.json", "w") as f:

    json.dump(
        splits,
        f,
        indent=4
    )


# --------------------------------------------------
# Display distributions
# --------------------------------------------------

print("\n----------------------------")
print("TRAINING SET")
print("----------------------------")

print(
    "Training images:",
    len(train_paths)
)

train_counts = Counter(train_labels)

for label, class_name in enumerate(class_names):

    print(
        f"{class_name}: "
        f"{train_counts[label]}"
    )


print("\n----------------------------")
print("VALIDATION SET")
print("----------------------------")

print(
    "Validation images:",
    len(val_paths)
)

val_counts = Counter(val_labels)

for label, class_name in enumerate(class_names):

    print(
        f"{class_name}: "
        f"{val_counts[label]}"
    )


print("\nSplit saved to splits.json")