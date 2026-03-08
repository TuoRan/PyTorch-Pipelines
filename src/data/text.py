# src/data/text.py
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.feature_extraction.text import TfidfVectorizer
from datasets import load_dataset

class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        return sample, label

def get_agnews_loaders(
    batch_size: int,
    num_workers: int,
    val_split: float = 0.1,
    max_features: int = 10000,
    seed: int = 42,
):
    """
    Load AG News, create a train/validation split from the original train split,
    fit TF-IDF on train only, and return DataLoaders plus the fitted vectorizer.

    Args:
        batch_size(int): DataLoader batch size.
        num_workers(int): Number of subprocesses for data loading.
        val_split(float): Fraction of original training split used for validation.
        max_features(int): Maximum vocabulary size for TF-IDF.
        seed(int): Random seed for reproducible train/val split.

    Returns:
        train_loader, val_loader, test_loader, vectorizer
    """
    data = load_dataset("ag_news")

    full_train_texts = data["train"]["text"]
    full_train_labels = data["train"]["label"]
    test_texts = data["test"]["text"]
    test_labels = data["test"]["label"]

    # Reproducible train/val split on the original training split
    n_train_total = len(full_train_texts)
    indices = np.arange(n_train_total)

    rng = np.random.default_rng(seed)
    rng.shuffle(indices)

    val_size = int(n_train_total * val_split)
    train_size = n_train_total - val_size

    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    train_texts = [full_train_texts[i] for i in train_indices]
    val_texts = [full_train_texts[i] for i in val_indices]

    train_labels = [full_train_labels[i] for i in train_indices]
    val_labels = [full_train_labels[i] for i in val_indices]

    # Define vectorizer
    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=2,
        max_features=max_features,
    )

    # Fit vectorizer on train set only
    X_train = vectorizer.fit_transform(train_texts)
    X_val = vectorizer.transform(val_texts)
    X_test = vectorizer.transform(test_texts)

    # Sparse matrix ––> dense ––> torch tensors
    X_train = torch.tensor(np.asarray(X_train.todense()), dtype=torch.float32)
    X_val = torch.tensor(np.asarray(X_val.todense()), dtype=torch.float32)
    X_test = torch.tensor(np.asarray(X_test.todense()), dtype=torch.float32)

    y_train = torch.tensor(train_labels, dtype=torch.long)
    y_val = torch.tensor(val_labels, dtype=torch.long)
    y_test = torch.tensor(test_labels, dtype=torch.long)

    train_data = CustomDataset(X_train, y_train)
    val_data = CustomDataset(X_val, y_val)
    test_data = CustomDataset(X_test, y_test)

    train_loader = DataLoader(
        dataset=train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        dataset=val_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        dataset=test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, val_loader, test_loader, vectorizer