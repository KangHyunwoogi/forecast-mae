import torch
import torch.nn as nn

class BinaryClassificationDecoder(nn.Module):
    """A simple MLP-based binary classification decoder"""

    def __init__(self, embed_dim) -> None:
        super().__init__()

        self.embed_dim = embed_dim

        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),  # Output layer for binary classification
            nn.Sigmoid()  # Sigmoid activation to output probability
        )

    def forward(self, x):
        return self.classifier(x)