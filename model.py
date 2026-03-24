"""
Small CNN for keyword spotting. Fast inference (~2-5ms on GPU, ~10ms on CPU).
"""

import torch
import torch.nn as nn
from audio_processing import N_MELS


class VoiceCommandCNN(nn.Module):
    def __init__(self, num_classes: int):
        """
        4-block CNN feature extractor followed by adaptive pooling classifier.

        Feature block shapes:
            Block 1: (1, 40, 101) -> (32, 20, 50)
            Block 2: (32, 20, 50) -> (64, 10, 25)
            Block 3: (64, 10, 25) -> (128, 5, 12)
            Block 4: (128, 5, 12) -> (128, 2, 6)

        @param num_classes: Number of output classes.
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
