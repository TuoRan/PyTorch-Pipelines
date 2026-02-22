# scripts/train_agnews.py
import torch
from torch import nn

from src.utils.device import get_device
from src.utils.seed import set_seed
from src.data.text import get_agnews_loaders
from src.models.registry import get_model
from src.training.loops import fit
from src.training.reporting import make_run_dir, save_json

def main():
    set_seed(123)
    device, info = get_device()
    print("Device:", info)

    train_loader, test_loader, vectorizer = get_agnews_loaders(batch_size=128, num_workers=3)
    
    # Define model
    model_name = "text_mlp"
    model = get_model(name=model_name, vocab_size=len(vectorizer.vocabulary_), num_classes=4, dropout_p=0.3, device=device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # Run metadata
    run_dir = make_run_dir("runs", f"agnews_{model_name}_seed123")

    config = {
        "dataset": "agnews",
        "model": {
            "name": model_name,
            "num_classes": 4,
        },
        "optimizer": {
            "name": "adamw",
            "lr": 1e-3,
        },
        "epochs": 3,
        "batch_size": 128,
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
        epochs=3,
        run_dir=run_dir,
        save_best=False,
        save_last=False,
        extra_ckpt_data=config,
        device=device
    )
    

if __name__ == "__main__":
    main()