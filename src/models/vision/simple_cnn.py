# src/models/vision/simple_cnn.py
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int, dropout_p: float):
        super(SimpleCNN, self).__init__()

        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.dropout = nn.Dropout(p=float(dropout_p))
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(in_features=128, out_features=num_classes)

    def forward(self, x):
        out = self.convblock1(x)
        out = self.convblock2(out)
        out = self.avgpool(out)

        out = self.dropout(out)
        out = self.flatten(out)
        out = self.linear(out)
        return out