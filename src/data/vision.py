import torch
from torchvision import datasets
import torchvision.transforms.v2 as transforms
from torch.utils.data import DataLoader

def get_cifar10_loaders(batch_size, num_workers):
    train_transform = transforms.Compose([
        transforms.Pad(padding=4),
        transforms.RandomCrop(size=(32, 32)),
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.ToTensor(),
        transforms.Normalize((0.491, 0.482, 0.447), (0.247, 0.243, 0.262))
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.491, 0.482, 0.447), (0.247, 0.243, 0.262))
    ])

    training_data = datasets.CIFAR10(
        root="data",
        train=True,
        download=True,
        transform=train_transform
    )

    test_data = datasets.CIFAR10(
        root="data",
        train=False,
        download=True,
        transform=test_transform
    )

    train_loader = DataLoader(dataset=training_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(dataset=test_data, batch_size=batch_size, num_workers=num_workers)

    return train_loader, test_loader