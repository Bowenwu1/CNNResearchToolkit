import os
import torch
import torchvision
import torchvision.transforms as transforms


def get_dataloaders(batch_size, n_workers, path=""):

    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
            ),
        ]
    )

    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
            ),
        ]
    )

    trainset = torchvision.datasets.CIFAR10(
        root=path, train=True, download=True, transform=transform_train
    )
    dataloader_train = torch.utils.data.DataLoader(
        trainset, batch_size, shuffle=True, num_workers=n_workers
    )

    testset = torchvision.datasets.CIFAR10(
        root=path, train=False, download=True, transform=transform_test
    )
    dataloader_test = torch.utils.data.DataLoader(
        testset, batch_size, shuffle=False, num_workers=n_workers
    )

    return dataloader_train, dataloader_test
