# src/models/vision/resnet18/resnet18.py
import torch.nn as nn
from src.models.vision.resnet.basic_block import BasicBlock

# Define ResNet-18 architecture using BasicBlock layers
class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1, bias=False)
        self.in_channels = 64
        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block=BasicBlock, channels=64, num_blocks=2, stride=1)
        self.layer2 = self._make_layer(block=BasicBlock, channels=128, num_blocks=2, stride=2)
        self.layer3 = self._make_layer(block=BasicBlock, channels=256, num_blocks=2, stride=2)
        self.layer4 = self._make_layer(block=BasicBlock, channels=512, num_blocks=2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(in_features=512, out_features=10)

    # Define method to create BasicBlock layers with possible downsampling and deeper representations
    # First block handles resolution & channel changes
    # Second block handles refines features with deeper representations
    def _make_layer(self, block, channels, num_blocks, stride):
        blocks = []
        blocks.append(block(self.in_channels, channels, stride))
        self.in_channels = channels
        for _ in range(num_blocks-1):
            blocks.append(block(self.in_channels, channels, stride=1))
        return nn.Sequential(*blocks)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn(out)
        out = self.relu(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avgpool(out)
        out = self.flatten(out)
        out = self.fc(out)
        return out