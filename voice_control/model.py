"""
4-block CNN for keyword spotting.

Kept small on purpose (~250k params) so each inference fits well
inside the 75ms audio stride — a slower model would back up the
audio thread and drop frames.
"""

import torch
import torch.nn as nn


class VoiceCommandCNN(nn.Module):
    def __init__(self, num_classes: int):
        """
        Conv stack over (1, 40, 101) log mel, then adaptive-pooled head.
        AdaptiveAvgPool2d(1) means the classifier is decoupled from
        spatial dims — conv stack can be retuned without touching
        the Linear layer.

        Shapes:
            Block 1: (1, 40, 101) -> (32, 20, 50)
            Block 2: (32, 20, 50) -> (64, 10, 25)
            Block 3: (64, 10, 25) -> (128, 5, 12)
            Block 4: (128, 5, 12) -> (128, 2, 6)

        @param num_classes: commands + _unknown + _silence.
        """
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x
