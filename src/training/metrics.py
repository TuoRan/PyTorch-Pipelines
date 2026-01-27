# src/training/metrics.py

def print_metrics(metrics):
    epoch = metrics.get('epoch', '')
    train_loss = metrics.get('train_loss', 0)
    test_loss = metrics.get('test_loss', 0)
    test_acc = metrics.get('test_acc', 0)

    print("\n" + "-" * 70)
    print(f"Epoch: {epoch} | Train Loss: {train_loss:.2f} | Test Loss: {test_loss:.2f} | Test Acc: {test_acc*100:.2f}%")
    print("-" * 70 + "\n")