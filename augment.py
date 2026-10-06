import sys
import torch
import matplotlib.pyplot as plt

from PIL import Image
from torchvision import transforms


class FaceAugmentor:

    def __init__(self):

        # Resize and convert image to tensor FIRST
        self.prepare = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

        # ------------------------------------------------
        # Six required augmentation techniques
        # ------------------------------------------------

        self.augmentations = {

            "Horizontal Flip":
                transforms.RandomHorizontalFlip(p=1.0),

            "Rotation":
                transforms.RandomRotation(degrees=(-15, 15)),

            "Brightness Jitter":
                transforms.ColorJitter(brightness=0.4),

            "Contrast Jitter":
                transforms.ColorJitter(contrast=0.4),

            "Gaussian Blur":
                transforms.GaussianBlur(
                    kernel_size=5,
                    sigma=(0.1, 2.0)
                ),

            "Perspective Transform":
                transforms.RandomPerspective(
                    distortion_scale=0.3,
                    p=1.0
                )
        }

        # Pipeline used when generating random augmentations
        self.random_augmentation = transforms.Compose([

            transforms.RandomHorizontalFlip(p=0.5),

            transforms.RandomRotation(
                degrees=(-15, 15)
            ),

            transforms.ColorJitter(
                brightness=0.3,
                contrast=0.3
            ),

            transforms.RandomApply(
                [
                    transforms.GaussianBlur(
                        kernel_size=5,
                        sigma=(0.1, 2.0)
                    )
                ],
                p=0.3
            ),

            transforms.RandomPerspective(
                distortion_scale=0.2,
                p=0.3
            )
        ])


    # ------------------------------------------------
    # Load image
    # ------------------------------------------------

    def load_image(self, image_path):

        image = Image.open(image_path).convert("RGB")

        # Convert to tensor
        image = self.prepare(image)

        return image


    # ------------------------------------------------
    # Return N random augmentations
    # ------------------------------------------------

    def get_random_augmentations(
        self,
        image_path,
        n=6
    ):

        image = self.load_image(image_path)

        augmented_images = []

        for _ in range(n):

            augmented = self.random_augmentation(
                image.clone()
            )

            augmented_images.append(augmented)

        return augmented_images


    # ------------------------------------------------
    # Visualize original + augmentations
    # ------------------------------------------------

    def visualize(self, image_path):

        original = self.load_image(image_path)

        images = [original]
        titles = ["Original"]

        for name, augmentation in self.augmentations.items():

            augmented = augmentation(
                original.clone()
            )

            images.append(augmented)
            titles.append(name)

        # 2 x 4 grid
        rows = 2
        cols = 4

        plt.figure(figsize=(14, 7))

        for i, image in enumerate(images):

            plt.subplot(
                rows,
                cols,
                i + 1
            )

            # PyTorch:
            # C x H x W
            #
            # Matplotlib:
            # H x W x C
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

            plt.imshow(image)

            plt.title(
                titles[i]
            )

            plt.axis("off")

        plt.tight_layout()

        plt.show()


# ------------------------------------------------
# Main
# ------------------------------------------------

if __name__ == "__main__":

    if len(sys.argv) != 2:

        print(
            "Usage: python augment.py <image_path>"
        )

        print(
            "Example:"
        )

        print(
            "python augment.py raw/navodhya/image.jpg"
        )

        sys.exit(1)


    image_path = sys.argv[1]

    augmentor = FaceAugmentor()


    # Show visualization grid
    augmentor.visualize(
        image_path
    )


    # Generate six random augmentations
    random_images = (
        augmentor.get_random_augmentations(
            image_path,
            n=6
        )
    )

    print(
        f"\nGenerated "
        f"{len(random_images)} "
        f"random augmentations."
    )

    for i, image in enumerate(
        random_images,
        start=1
    ):

        print(
            f"Augmentation {i}: "
            f"shape={image.shape}, "
            f"dtype={image.dtype}"
        )