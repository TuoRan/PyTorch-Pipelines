# src/training/loops.py
import torch
from src.training.metrics import print_metrics

def train_one_epoch(model, optimizer, criterion, train_loader, device):
    model.train()
    train_loss = 0.0
    n_batches = 0

    for X, y in train_loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()

        y_pred = model(X)
        loss = criterion(y_pred, y)

        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        n_batches += 1

    return train_loss / max(n_batches, 1)

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

def fit(model, optimizer, criterion, train_loader, test_loader, device, epochs):
    # Initiate dict for metrics and list for epoch metric history
    metric_dict = {'epoch' : int, 'train_loss' : float, 'test_loss' : float, 'test_acc' : float}
    history = []

    for epoch in range(epochs):
        print(f"Training epoch: {epoch+1}/{epochs}...")
        train_loss = train_one_epoch(model, optimizer, criterion, train_loader, device)
        test_loss, test_acc = evaluate(model, criterion, test_loader, device)

        # Capture metrics and append into history
        metric_dict = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'test_loss': test_loss,
            'test_acc': test_acc
        }
        history.append(metric_dict)

        # Print metrics
        print_metrics(metric_dict)

    return history