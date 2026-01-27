# scripts/train_cifar10.py
import torch
from torch import nn

from src.utils.device import get_device
from src.utils.seed import set_seed
from src.training.loops import fit
from src.models.vision.simple_cnn import SimpleCNN
from src.data.vision import get_cifar10_loaders

def main():
    torch.set_warn_always(False)
    set_seed(123)
    device, info = get_device()
    print("Device:", info)

    train_loader, test_loader = get_cifar10_loaders(batch_size=64, num_workers=3)

    # Define model
    model = SimpleCNN(num_classes=10).to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    # Train and evaluate model, print metrics
    history = fit(model, optimizer, criterion, train_loader, test_loader, device, epochs=2)
    for i in history:
        print(i)

if __name__ == "__main__":
    main()