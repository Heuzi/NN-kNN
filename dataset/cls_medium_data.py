import torch
from torchvision import datasets, transforms
import os

folder_path = "./dataset/"

def compute_mean_std(dataset):
    loader = torch.utils.data.DataLoader(dataset, batch_size=1024, num_workers=2)
    mean = 0.0
    std = 0.0
    total_samples = 0

    for images, _ in loader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, -1)

        mean += images.mean(1).sum(0)
        std += images.std(1).sum(0)
        total_samples += batch_samples

    mean /= total_samples
    std /= total_samples

    return mean.item(), std.item()


def MNIST():
    initial_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    train_dataset = datasets.MNIST(root=folder_path, train=True, download=True, transform=initial_transform)
    test_dataset = datasets.MNIST(root=folder_path, train=False, download=True, transform=initial_transform)

    mean, std = compute_mean_std(train_dataset)
    print(f"Computed Mean: {mean:.4f}, Computed Std: {std:.4f}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((mean,), (std,))
    ])

    train_dataset = datasets.MNIST(root=folder_path, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root=folder_path, train=False, download=True, transform=transform)

    X_train = torch.stack([train_dataset[i][0] for i in range(len(train_dataset))])
    y_train = torch.tensor([train_dataset[i][1] for i in range(len(train_dataset))])

    X_test = torch.stack([test_dataset[i][0] for i in range(len(test_dataset))])
    y_test = torch.tensor([test_dataset[i][1] for i in range(len(test_dataset))])

    return X_train, y_train, X_test, y_test


def CIFAR10(download_path=folder_path):
    initial_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    train_dataset = datasets.CIFAR10(root=download_path, train=True,
                                                 download=True, transform=initial_transform)
    test_dataset = datasets.CIFAR10(root=download_path, train=False,
                                                download=True, transform=initial_transform)

    mean, std = compute_mean_std(train_dataset)
    print(f"Computed Mean: {mean}, Computed Std: {std}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    train_dataset = datasets.CIFAR10(root=download_path, train=True,
                                                 download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root=download_path, train=False,
                                                download=True, transform=transform)

    X_train = torch.stack([train_dataset[i][0] for i in range(len(train_dataset))])
    y_train = torch.tensor([train_dataset[i][1] for i in range(len(train_dataset))])

    X_test = torch.stack([test_dataset[i][0] for i in range(len(test_dataset))])
    y_test = torch.tensor([test_dataset[i][1] for i in range(len(test_dataset))])

    classes = train_dataset.classes

    return X_train, y_train, X_test, y_test

def SVHN(download_path=folder_path):

    download_path = os.path.join(download_path, 'svhn')
    initial_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    train_dataset = datasets.SVHN(root=download_path, split='train',
                                                 download=True, transform=initial_transform)
    test_dataset = datasets.SVHN(root=download_path, split='test',
                                                download=True, transform=initial_transform)

    mean, std = compute_mean_std(train_dataset)
    print(f"Computed Mean: {mean}, Computed Std: {std}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    train_dataset = datasets.SVHN(root=download_path, split='train', transform=transform)
    test_dataset = datasets.SVHN(root=download_path, split='test', transform=transform)

    X_train = torch.stack([train_dataset[i][0] for i in range(len(train_dataset))])
    y_train = torch.tensor([train_dataset[i][1] for i in range(len(train_dataset))])

    X_test = torch.stack([test_dataset[i][0] for i in range(len(test_dataset))])
    y_test = torch.tensor([test_dataset[i][1] for i in range(len(test_dataset))])

    return X_train, y_train, X_test, y_test

def ImageNet32(download_path=folder_path):
    pass

def TinyImageNet(download_path=folder_path):
    pass

def SST1():
    # !pip install datasets
    from datasets import load_dataset
    #dummy system, because training uses cfg somewhere and I am too lazy to change

    # Load SST-1 dataset
    dataset = load_dataset("sst", "default")

    # Map continuous labels to discrete classes
    def _map_to_classes(example):
        if example['label'] < 0.2:
            example['label'] = 0  # Very Negative
        elif example['label'] < 0.4:
            example['label'] = 1  # Negative
        elif example['label'] < 0.6:
            example['label'] = 2  # Neutral
        elif example['label'] < 0.8:
            example['label'] = 3  # Positive
        else:
            example['label'] = 4  # Very Positive
        return example

    # Apply mapping
    dataset = dataset.map(_map_to_classes)

    # Access train, validation, and test splits
    train_data = dataset['train']
    val_data = dataset['validation']
    test_data = dataset['test']

    return train_data, val_data, test_data

def SST2():
    from datasets import load_dataset

    # Load SST-1 dataset
    dataset = load_dataset("sst", "default")

    # Map continuous labels to binary classes: Negative (0, 1) and Positive (3, 4)
    def _map_to_binary_classes(example):
        if example['label'] < 0.4:  # Negative (Very Negative or Negative)
            example['label'] = 0  # Negative
        elif example['label'] >= 0.6:  # Positive (Positive or Very Positive)
            example['label'] = 1  # Positive
        else:
            return None  # Exclude Neutral (labels between 0.4 and 0.6)
        return example

    # Apply mapping and remove neutral examples
    dataset = dataset.filter(lambda example: example['label'] < 0.4 or example['label'] >= 0.6)
    dataset = dataset.map(_map_to_binary_classes)

    # Access train, validation, and test splits
    train_data = dataset['train']
    val_data = dataset['validation']
    test_data = dataset['test']

    # Example: Print first training example
    return train_data, val_data, test_data

def CoLA(download_path=folder_path):
    pass

def AGNews(download_path=folder_path):
    pass

DATATYPES = {
    'mnist': MNIST,
    'cifar10': CIFAR10,
    'svhn': SVHN,
    'imagenet32': ImageNet32,
    'tinyimagenet': TinyImageNet, 
    'sst1': SST1,
    'sst2': SST2,
    'cola': CoLA,
    'agnews': AGNews,
}

def Cls_medium_data(dataset):
    package = DATATYPES[dataset]()
    return package
