# src/data/vision.py
import torch
from torchvision import datasets
import torchvision.transforms.v2 as transforms
from torch.utils.data import DataLoader, random_split, Subset

def get_cifar10_loaders(batch_size: int, num_workers: int, val_split: float = 0.1):
    """
    Transformations and data loaders for CIFAR-10
    
    Args:
        batch_size(int): Data loader batch size.
        num_workers(int): How many subprocesses to use for data loading.
        val_split(float): Define validation split size (percentile of the whole dataset).

    Returns:
        (train_loader, val_loader, test_loader) dataloaders for training, validation, and testing sets
    """
    train_transform = transforms.Compose([
        transforms.Pad(padding=4),
        transforms.RandomCrop(size=(32, 32)),
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.ToImage(), 
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize((0.491, 0.482, 0.447), (0.247, 0.243, 0.262))
    ])

    val_transform = transforms.Compose([
        transforms.ToImage(), 
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize((0.491, 0.482, 0.447), (0.247, 0.243, 0.262))
    ])

    test_transform = val_transform

    full_dataset = datasets.CIFAR10(
        root="data",
        train=True,
        download=True
    )

    train_size = int(len(full_dataset) * (1 - val_split))
    val_size = len(full_dataset) - train_size

    generator = torch.Generator().manual_seed(42)
    train_subset, val_subset = random_split(full_dataset, [train_size, val_size], generator=generator)

    train_data = datasets.CIFAR10(
        root="data",
        train=True,
        download=False,
        transform=train_transform
    )

    val_data = datasets.CIFAR10(
        root="data",
        train=True,
        download=False,
        transform=val_transform
    )
    
    train_data = Subset(train_data, train_subset.indices)
    val_data = Subset(val_data, val_subset.indices)

    test_data = datasets.CIFAR10(
        root="data",
        train=False,
        download=False,
        transform=test_transform
    )

    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(dataset=val_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader