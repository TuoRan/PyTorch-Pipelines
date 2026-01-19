import torch

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
    n_batches = 0

    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)

            y_pred = model(X)
            loss = criterion(y_pred, y)

            test_loss += loss.item()
            n_batches += 1
            
    return test_loss / max(n_batches,1)