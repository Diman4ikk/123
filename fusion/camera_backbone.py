"""ResNet-50 image feature extractor for the LiDAR-camera fusion project."""


from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image
from torch import Tensor, nn
from torchvision import transforms
from torchvision.models import ResNet50_Weights, resnet50


def load_image_tensor(
    image_path: Path,
    weights: ResNet50_Weights,
) -> Tensor:
    """Load an RGB image and return a normalized tensor [1, 3, H, W]."""
    image_path = Path(image_path)

    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    with Image.open(image_path) as img:
        image = img.convert("RGB")

    # Get normalization parameters without resizing or cropping the image.
    preprocessing = weights.transforms()

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=preprocessing.mean,
                std=preprocessing.std,
            ),
        ]
    )

    return transform(image).unsqueeze(0)


def build_backbone(
    weights: ResNet50_Weights = ResNet50_Weights.DEFAULT,
) -> nn.Module:
    """Build ResNet-50 without global pooling and the classification head."""
    model = resnet50(weights=weights)

    # Keep the convolutional feature extractor only.
    backbone = nn.Sequential(*list(model.children())[:-2])
    

    return backbone


class CameraBackbone(nn.Module):
    """ResNet-50 feature extractor as a standard PyTorch module.

    Input:  normalized batch [B, 3, H, W]
    Output: spatial feature map [B, 2048, H', W']
    """

    def __init__(self, weights: ResNet50_Weights = ResNet50_Weights.DEFAULT) -> None:
        super().__init__()
        self.weights = weights
        self.backbone = build_backbone(weights)

    def forward(self, images: Tensor) -> Tensor:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(
                f"Expected images shaped [B, 3, H, W], got {tuple(images.shape)}"
            )
        return self.backbone(images)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, help="Path to one nuScenes camera image")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CameraBackbone().to(device)
    model.eval()
    image = load_image_tensor(args.image, model.weights).to(device)
    with torch.inference_mode():
        features = model(image).cpu()

    print("Model device:", device)
    print("Feature shape:", tuple(features.shape))
    print("Features finite:", bool(torch.isfinite(features).all()))
    print("Feature range:", float(features.min()), float(features.max()))
    print("Output device:", features.device)


if __name__ == "__main__":
    main()