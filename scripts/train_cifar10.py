# scripts/train_cifar10.py
import torch
from torch import nn

from src.utils.device import get_device
from src.utils.seed import set_seed
from src.data.vision import get_cifar10_loaders
from src.models.registry import get_model
from src.training.loops import fit
from src.training.reporting import make_run_dir, save_json

def main():
    torch.set_warn_always(False)
    set_seed(123)
    device, info = get_device()
    print("Device:", info)

    train_loader, test_loader = get_cifar10_loaders(batch_size=64, num_workers=3)

    # Define model
    model_name = "simple_cnn"
    model = get_model(name=model_name, dropout_p=0.3, device=device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    # Run metadata
    run_dir = make_run_dir("runs", f"cifar10_{model_name}_seed123")

    config = {
        "dataset": "cifar10",
        "model": {
            "name": model_name,
            "num_classes": 10,
        },
        "optimizer": {
            "name": "sgd",
            "lr": 0.1,
        },
        "epochs": 4,
        "batch_size": 64,
        "seed": 123,
        "device": info,
    }
    save_json(f"{run_dir}/config.json", config)

    # Train and evaluate model, print metrics, save metrics/checkpoints
    history = fit(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        train_loader=train_loader,
        test_loader=test_loader,
        epochs=6,
        run_dir=run_dir,
        save_best=False,
        save_last=False,
        extra_ckpt_data=config,
        device=device
    )
    

if __name__ == "__main__":
    main()