# scripts/smoke_train_loop.py
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.utils.device import get_device
from src.utils.seed import set_seed
from src.training.loops import train_one_epoch, evaluate


def main():
    set_seed(123)
    device, info = get_device()
    print("Device:", info)

    # Tiny fake classification dataset
    X = torch.randn(256, 3, 32, 32)
    y = torch.randint(0, 10, (256,))

    train_loader = DataLoader(TensorDataset(X, y), batch_size=64, shuffle=True)
    test_loader = DataLoader(TensorDataset(X, y), batch_size=64, shuffle=False)

    # Tiny model that matches CIFAR-like shape
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(3 * 32 * 32, 64),
        nn.ReLU(),
        nn.Linear(64, 10),
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    # Check params change after training
    before = model[1].weight.detach().clone()

    train_loss = train_one_epoch(model, optimizer, criterion, train_loader, device)
    test_loss = evaluate(model, criterion, test_loader, device)

    after = model[1].weight.detach().clone()

    print(f"train_loss: {train_loss:.4f} | test_loss: {test_loss:.4f}")
    print("params_changed:", not torch.equal(before, after))
    assert torch.isfinite(torch.tensor(train_loss)), "Train loss is not finite"
    assert not torch.equal(before, after), "Parameters did not update"


if __name__ == "__main__":
    main()
