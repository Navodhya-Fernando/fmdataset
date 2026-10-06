import time
import torch
import torch.nn as nn

from torchvision.models import mobilenet_v2
from dataloader import create_dataloaders


# --------------------------------------------------
# Settings
# --------------------------------------------------

NUM_CLASSES = 4
EPOCHS = 3
LEARNING_RATE = 0.001

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)


# --------------------------------------------------
# Train and evaluate one batch size
# --------------------------------------------------

def run_experiment(batch_size):

    print("\n" + "=" * 50)
    print(f"BATCH SIZE = {batch_size}")
    print("=" * 50)

    # Create fresh DataLoaders
    train_loader, val_loader, class_names = (
        create_dataloaders(
            ".",
            batch_size=batch_size
        )
    )

    # --------------------------------------------------
    # MobileNet V2
    # --------------------------------------------------

    model = mobilenet_v2(
        weights=None
    )

    # Change final layer for our 4 classes
    model.classifier[1] = nn.Linear(
        model.last_channel,
        NUM_CLASSES
    )

    model = model.to(device)


    # --------------------------------------------------
    # Loss + optimizer
    # --------------------------------------------------

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )


    # --------------------------------------------------
    # Approximate input batch memory
    # --------------------------------------------------

    sample_images, _ = next(
        iter(train_loader)
    )

    batch_memory_mb = (
        sample_images.nelement()
        * sample_images.element_size()
        / (1024 ** 2)
    )

    print(
        f"Input batch memory: "
        f"{batch_memory_mb:.2f} MB"
    )

    print(
        "Training batches:",
        len(train_loader)
    )

    print(
        "Validation batches:",
        len(val_loader)
    )


    # Reset GPU memory statistics if CUDA exists
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


    # --------------------------------------------------
    # TRAINING
    # --------------------------------------------------

    start_time = time.time()

    final_train_loss = 0.0

    for epoch in range(EPOCHS):

        model.train()

        running_loss = 0.0
        correct = 0
        total = 0


        for images, labels in train_loader:

            images = images.to(device)
            labels = labels.to(device)


            optimizer.zero_grad()


            outputs = model(images)


            loss = criterion(
                outputs,
                labels
            )


            loss.backward()

            optimizer.step()


            running_loss += (
                loss.item()
                * images.size(0)
            )


            _, predicted = torch.max(
                outputs,
                1
            )


            total += labels.size(0)

            correct += (
                predicted == labels
            ).sum().item()


        epoch_loss = (
            running_loss
            / total
        )

        epoch_accuracy = (
            100
            * correct
            / total
        )

        final_train_loss = epoch_loss


        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"Loss: {epoch_loss:.4f} | "
            f"Train Accuracy: "
            f"{epoch_accuracy:.2f}%"
        )


    training_time = (
        time.time()
        - start_time
    )


    # --------------------------------------------------
    # VALIDATION
    # --------------------------------------------------

    model.eval()

    val_correct = 0
    val_total = 0
    val_loss = 0.0


    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)


            outputs = model(images)


            loss = criterion(
                outputs,
                labels
            )


            val_loss += (
                loss.item()
                * images.size(0)
            )


            _, predicted = torch.max(
                outputs,
                1
            )


            val_total += (
                labels.size(0)
            )


            val_correct += (
                predicted == labels
            ).sum().item()


    average_val_loss = (
        val_loss
        / val_total
    )


    val_accuracy = (
        100
        * val_correct
        / val_total
    )


    # --------------------------------------------------
    # Peak GPU memory
    # --------------------------------------------------

    if torch.cuda.is_available():

        peak_memory_mb = (
            torch.cuda.max_memory_allocated()
            / (1024 ** 2)
        )

    else:

        peak_memory_mb = None


    print("\nResults")

    print(
        f"Training time: "
        f"{training_time:.2f} seconds"
    )

    print(
        f"Final training loss: "
        f"{final_train_loss:.4f}"
    )

    print(
        f"Validation loss: "
        f"{average_val_loss:.4f}"
    )

    print(
        f"Validation accuracy: "
        f"{val_accuracy:.2f}%"
    )


    if peak_memory_mb is not None:

        print(
            f"Peak GPU memory: "
            f"{peak_memory_mb:.2f} MB"
        )

    else:

        print(
            "Peak GPU memory: "
            "N/A - training on CPU"
        )


    return {
        "batch_size": batch_size,
        "input_memory_mb": batch_memory_mb,
        "training_time": training_time,
        "train_loss": final_train_loss,
        "val_loss": average_val_loss,
        "val_accuracy": val_accuracy,
        "peak_memory_mb": peak_memory_mb
    }


# ==================================================
# RUN BOTH EXPERIMENTS
# ==================================================

results_8 = run_experiment(
    batch_size=8
)

results_64 = run_experiment(
    batch_size=64
)


# ==================================================
# FINAL COMPARISON
# ==================================================

print("\n")
print("=" * 70)
print("FINAL BATCH SIZE COMPARISON")
print("=" * 70)


print(
    f"{'Aspect':<25}"
    f"{'Batch 8':<20}"
    f"{'Batch 64':<20}"
)


print("-" * 70)


print(
    f"{'Input memory (MB)':<25}"
    f"{results_8['input_memory_mb']:<20.2f}"
    f"{results_64['input_memory_mb']:<20.2f}"
)


print(
    f"{'Training time (sec)':<25}"
    f"{results_8['training_time']:<20.2f}"
    f"{results_64['training_time']:<20.2f}"
)


print(
    f"{'Final train loss':<25}"
    f"{results_8['train_loss']:<20.4f}"
    f"{results_64['train_loss']:<20.4f}"
)


print(
    f"{'Validation loss':<25}"
    f"{results_8['val_loss']:<20.4f}"
    f"{results_64['val_loss']:<20.4f}"
)


print(
    f"{'Validation accuracy':<25}"
    f"{results_8['val_accuracy']:<20.2f}"
    f"{results_64['val_accuracy']:<20.2f}"
)


print("\nExperiment complete.")