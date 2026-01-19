import torch
from torch.utils.data import DataLoader
import numpy as np
from src.data.vision import get_cifar10_loaders

def calculate_data_stats(dataset):
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=1)
    mean = 0.0
    std = 0.0
    total_samples = 0
    
    for images, _ in loader:
        # Rearrange batch to be (batch_size * num_channels, height, width) for mean/std calculation
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_samples += batch_samples

    mean /= total_samples
    std /= total_samples
    return mean, std

# Calculate dataset statistics
train_dataset = get_cifar10_loaders(batch_size=64, num_workers=1)
mean, std = calculate_data_stats(train_dataset)
print(f'Mean: {mean}, Std: {std}')