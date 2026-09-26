import torch.nn as nn
from torchvision.models import (
    efficientnet_b0,
    EfficientNet_B0_Weights,
)


class DermatologyEfficientNet(nn.Module):

    def __init__(
        self,
        num_classes=6,
        freeze_backbone=True,
    ):
        super().__init__()

        weights = EfficientNet_B0_Weights.DEFAULT

        self.model = efficientnet_b0(
            weights=weights
        )

        # Replace ImageNet classifier
        in_features = self.model.classifier[1].in_features

        self.model.classifier[1] = nn.Linear(
            in_features,
            num_classes
        )

        # Freeze backbone only when requested
        if freeze_backbone:

            for param in self.model.features.parameters():
                param.requires_grad = False

    def forward(self, x):
        return self.model(x)