"""Image preprocessing utilities."""
import io
import base64
from typing import Tuple
from PIL import Image
import torch
import torchvision.transforms as transforms
import numpy as np

from config import CIFAR_MEAN, CIFAR_STD, INPUT_SIZE


class ImagePreprocessor:
    """Handles image preprocessing for CIFAR-style inputs."""

    def __init__(self):
        """Initialize preprocessor with CIFAR normalization."""
        self.transform = transforms.Compose([
            transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=CIFAR_MEAN, std=CIFAR_STD)
        ])

        # For creating preview images (no normalization)
        self.preview_transform = transforms.Compose([
            transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        ])

    def preprocess_image(self, image: Image.Image) -> Tuple[torch.Tensor, str]:
        """Preprocess a single image.

        Args:
            image: PIL Image

        Returns:
            Tuple of (preprocessed tensor, base64 preview string)
        """
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Create preview
        preview_img = self.preview_transform(image)
        preview_base64 = self._image_to_base64(preview_img)

        # Preprocess for model
        tensor = self.transform(image)

        return tensor, preview_base64

    def preprocess_batch(self, images: list) -> Tuple[torch.Tensor, list]:
        """Preprocess a batch of images.

        Args:
            images: List of PIL Images

        Returns:
            Tuple of (batched tensor [N, 3, 32, 32], list of base64 previews)
        """
        tensors = []
        previews = []

        for img in images:
            tensor, preview = self.preprocess_image(img)
            tensors.append(tensor)
            previews.append(preview)

        # Stack into batch
        batch_tensor = torch.stack(tensors)

        return batch_tensor, previews

    @staticmethod
    def _image_to_base64(image: Image.Image) -> str:
        """Convert PIL Image to base64 string.

        Args:
            image: PIL Image

        Returns:
            Base64 encoded string with data URI prefix
        """
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return f"data:image/png;base64,{img_str}"

    @staticmethod
    def validate_image(file_bytes: bytes) -> bool:
        """Validate if bytes represent a valid image.

        Args:
            file_bytes: Raw file bytes

        Returns:
            True if valid image, False otherwise
        """
        try:
            img = Image.open(io.BytesIO(file_bytes))
            img.verify()
            return True
        except Exception:
            return False
