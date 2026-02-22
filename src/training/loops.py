# src/training/loops.py
from __future__ import annotations
import torch
import os
from typing import Any, Dict, List, Optional, Union
from typing import Optional

from src.training.metrics import print_metrics
from src.training.reporting import save_json, save_checkpoint

def train_one_epoch(model, optimizer, criterion, train_loader, device):
    model.train()
    train_loss = 0.0
    train_acc = 0.0
    n_batches = 0

    for X, y in train_loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()

        y_pred = model(X)
        loss = criterion(y_pred, y)

        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        train_acc += torch.sum(torch.argmax(y_pred, dim=1) == y).item()
        n_batches += 1

    return train_loss / max(n_batches, 1), train_acc / len(train_loader.dataset)

def evaluate(model, criterion, test_loader, device):
    model.eval()
    test_loss = 0.0
    test_acc = 0.0
    n_batches = 0

    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)

            y_pred = model(X)
            loss = criterion(y_pred, y)

            test_loss += loss.item()
            test_acc += torch.sum(torch.argmax(y_pred, dim=1) == y).item()
            n_batches += 1
            
    return test_loss / max(n_batches,1), test_acc / len(test_loader.dataset)


def fit(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    train_loader,
    test_loader,
    epochs: int,
    run_dir: Optional[str] = None,
    save_best: bool = True,
    save_last: bool = True,
    extra_ckpt_data: Optional[Dict[str, Any]] = None,
    *,
    device: Union[str, torch.device],
) -> List[Dict[str, float]]:
    """
    Train for `epochs`, evaluate each epoch, and optionally persist metrics/checkpoints.

    If `run_dir` is provided:
      - metrics.json is updated each epoch
      - best.pt is saved on best test accuracy (if save_best=True)
      - last.pt is saved after training (if save_last=True)

    Returns:
      history: list of per-epoch metric dicts
    """
    dev = torch.device(device)

    if run_dir is not None:
        os.makedirs(run_dir, exist_ok=True)

    history: List[Dict[str, float]] = []
    best_test_acc: float = float("-inf")
    best_epoch: int = -1

    for epoch in range(epochs):
        epoch_idx = epoch + 1
        print(f"Training {model.__class__.__name__} | Epoch: {epoch_idx}/{epochs}...")

        train_loss, train_acc = train_one_epoch(model, optimizer, criterion, train_loader, dev)
        test_loss, test_acc = evaluate(model, criterion, test_loader, dev)

        metrics: Dict[str, float] = {
            "epoch": float(epoch_idx),
            "train_loss": float(train_loss),
            "train_acc": float(train_acc),
            "test_loss": float(test_loss),
            "test_acc": float(test_acc),
        }
        history.append(metrics)

        print_metrics(metrics)

        # Write metrics after every epoch (crash-safe)
        if run_dir is not None:
            save_json(
                os.path.join(run_dir, "metrics.json"),
                {
                    "history": history,
                    "best_test_acc": float(best_test_acc if best_epoch != -1 else test_acc),
                    "best_epoch": int(best_epoch if best_epoch != -1 else epoch_idx),
                },
            )

        # Save best checkpoint
        if run_dir is not None and save_best and test_acc > best_test_acc:
            best_test_acc = float(test_acc)
            best_epoch = epoch_idx
            save_checkpoint(
                os.path.join(run_dir, "best.pt"),
                model=model,
                optimizer=optimizer,
                epoch=epoch_idx,
                best_val_acc=best_test_acc,  # keep your helper name; it's just a float
                config=extra_ckpt_data or {},  # or pass your real config dict
            )

    # Save last checkpoint
    if run_dir is not None and save_last:
        # ensure best fields are sensible even if save_best=False
        if best_epoch == -1 and history:
            best_test_acc = float(history[-1]["test_acc"])
            best_epoch = int(history[-1]["epoch"])

        save_checkpoint(
            os.path.join(run_dir, "last.pt"),
            model=model,
            optimizer=optimizer,
            epoch=epochs,
            best_val_acc=float(best_test_acc),
            config=extra_ckpt_data or {},
        )

    return history