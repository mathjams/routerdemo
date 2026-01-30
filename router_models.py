import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    def __init__(
        self,
        num_classes=100,
        base_channels=32,
        num_layers=3,
        drop2d=0.1,   # conv-feature dropout
        drop=0.3      # classifier dropout
    ):
        super().__init__()
        layers = [
            nn.Conv2d(3, base_channels, 7, 2, 3),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(True),
            nn.Dropout2d(p=drop2d),

            nn.MaxPool2d(3, 2, 1),
        ]
        in_ch = base_channels
        for i in range(num_layers - 1):
            out_ch = base_channels * (2 ** (i + 1))
            layers.extend([
                nn.Conv2d(in_ch, out_ch, 3, 2, 1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(True),
                nn.Dropout2d(p=drop2d),
            ])
            in_ch = out_ch

        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=drop)   # after flatten
        self.classifier = nn.Linear(in_ch, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.classifier(x)

class RouterModel(nn.Module):
    def __init__(self, num_branches: int, drop2d=0.1, drop=0.2):
        super().__init__()
        c1, c2, c3, c4 = 64, 128, 256, 384  # ~1.26M params for 5 branches

        self.features = nn.Sequential(
            nn.Conv2d(3,  c1, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=drop2d),

            nn.Conv2d(c1, c2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=drop2d),

            nn.Conv2d(c2, c3, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c3),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=drop2d),

            nn.Conv2d(c3, c4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c4),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=drop2d),

            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.dropout = nn.Dropout(p=drop)
        self.classifier = nn.Linear(c4, num_branches)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.classifier(x)

