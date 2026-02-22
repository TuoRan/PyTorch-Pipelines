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

def get_agnews_loaders(batch_size: int, num_workers: int):

    data = load_dataset("ag_news")
    train_texts = data["train"]["text"]
    test_texts = data["test"]["text"]

    vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1,2), min_df=2, max_features=10000)
    X_train = vectorizer.fit_transform(train_texts)
    X_test = vectorizer.transform(test_texts)

    X_train = torch.tensor(np.asarray(X_train.todense()), dtype=torch.float32)
    X_test = torch.tensor(np.asarray(X_test.todense()), dtype=torch.float32)
    y_train = torch.tensor(data["train"]["label"], dtype=torch.long)
    y_test = torch.tensor(data["test"]["label"], dtype=torch.long)

    train_data = CustomDataset(X_train, y_train)
    test_data = CustomDataset(X_test, y_test)

    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(dataset=test_data, batch_size=batch_size, num_workers=num_workers)

    return train_loader, test_loader, vectorizer