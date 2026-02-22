# src/models/text/text_mlp.py
import torch.nn as nn

class TextMLP(nn.Module):
    def __init__(self, vocab_size: int, num_classes: int, dropout_p: float):
        super(TextMLP, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Linear(vocab_size, 512),
            nn.ReLU(),
            nn.Dropout(dropout_p)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout_p)
        )
        self.layer3 = nn.Linear(256, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        return out