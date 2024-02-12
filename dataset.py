import os
import random
from re import I
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Subset, DataLoader
from torchvision import datasets
from torchvision import transforms
from torchvision.datasets.utils import download_and_extract_archive, extract_archive
from torchvision.datasets import CIFAR10, CIFAR100, SVHN, MNIST


CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
CIFAR_STD = [0.2023, 0.1994, 0.2010]
general_test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
])


class CUTOUT(object):

    def __init__(self, length):
        self.length = length

    def __repr__(self):
        return ('{name}(length={length})'.format(name=self.__class__.__name__, **self.__dict__))

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


def get_cifar_test_loader(path="./dataset/", batch_size=64, num_workers=2):
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    test_data = datasets.CIFAR10(root=path, train=False, download=True, transform=test_transform)
    test_queue_cifar10 = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=num_workers)
    return test_queue_cifar10


def cifar10_dataloaders(batch_size=128, data_dir='./dataset/', num_workers=2, aug=False, cutout=-1, flatten=False, resize=None, crossval=False):

    test_transform_list = [
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD)
    ]
    if flatten:
        test_transform_list.append(transforms.Lambda(lambda x: torch.flatten(x)))
    if resize:
        test_transform_list.append(transforms.Resize((resize, resize)))
    test_transform = transforms.Compose(test_transform_list)

    if aug:
        train_transform_list = [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD)
        ]
        if cutout > 0 : train_transform_list += [CUTOUT(cutout)]
        if flatten:
            train_transform_list.append(transforms.Lambda(lambda x: torch.flatten(x)))
        if resize:
            train_transform_list.append(transforms.Resize((resize, resize)))
    else:
        train_transform_list = list(test_transform_list)
    train_transform = transforms.Compose(train_transform_list)

    train_set = CIFAR10(data_dir, train=True, transform=train_transform, download=False)
    if crossval:
        train_set = Subset(train_set, list(range(45000)))
        val_set = Subset(CIFAR10(data_dir, train=True, transform=test_transform, download=False), list(range(45000, 50000))) #  check random order
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    else:
        val_loader = None
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

    test_set = CIFAR10(data_dir, train=False, transform=test_transform, download=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader


def cifar100_dataloaders(batch_size=128, data_dir='./dataset/', num_workers=2, aug=False, cutout=-1, flatten=False, resize=None, crossval=False):

    test_transform_list = [
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD)
    ]
    if flatten:
        test_transform_list.append(transforms.Lambda(lambda x: torch.flatten(x)))
    if resize:
        test_transform_list.append(transforms.Resize((resize, resize)))
    test_transform = transforms.Compose(test_transform_list)

    if aug:
        train_transform_list = [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD)
        ]
        if cutout > 0 : train_transform_list += [CUTOUT(cutout)]
        if flatten:
            train_transform_list.append(transforms.Lambda(lambda x: torch.flatten(x)))
        if resize:
            train_transform_list.append(transforms.Resize((resize, resize)))
    else:
        train_transform_list = list(test_transform_list)
    train_transform = transforms.Compose(train_transform_list)

    train_set = CIFAR100(data_dir, train=True, transform=train_transform, download=False)
    if crossval:
        train_set = Subset(train_set, list(range(45000)))
        val_set = Subset(CIFAR100(data_dir, train=True, transform=test_transform, download=False), list(range(45000, 50000)))
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    else:
        val_loader = None
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

    test_set = CIFAR100(data_dir, train=False, transform=test_transform, download=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader


from DownsampledImageNet import ImageNet16
def imagenet_16_120_dataloaders(batch_size=128, data_dir='./dataset/', num_workers=2, aug=False, cutout=-1, flatten=False, resize=None):
    mean = [x / 255 for x in [122.68, 116.66, 104.01]]
    std  = [x / 255 for x in [63.22,  61.26 , 65.09]]

    test_transform_list = [
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]
    if flatten:
        test_transform_list.append(transforms.Lambda(lambda x: torch.flatten(x)))
    if resize:
        test_transform_list.append(transforms.Resize((resize, resize)))
    test_transform = transforms.Compose(test_transform_list)

    if aug:
        train_transform_list = [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(16, padding=2),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]
        if cutout > 0 : train_transform_list += [CUTOUT(cutout)]
        if flatten:
            train_transform_list.append(transforms.Lambda(lambda x: torch.flatten(x)))
        if resize:
            train_transform_list.append(transforms.Resize((resize, resize)))
    else:
        train_transform_list = list(test_transform_list)
    train_transform = transforms.Compose(train_transform_list)

    train_data = ImageNet16(data_dir, True , train_transform, 120)
    test_data  = ImageNet16(data_dir, False, test_transform , 120)
    assert len(train_data) == 151700 and len(test_data) == 6000

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, None, test_loader


def imagenet_dataloaders(batch_size=128, img_shape=64, data_dir='./dataset/', num_workers=2, aug=False, flatten=False, distributed=False):
    traindir = os.path.join(data_dir, 'train')
    validdir = os.path.join(data_dir, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    test_transform_list = [
        transforms.Resize(img_shape),
        transforms.CenterCrop(img_shape),
        transforms.ToTensor(),
        normalize,
    ]
    if flatten:
        test_transform_list.append(transforms.Lambda(lambda x: torch.flatten(x)))
    test_transform = transforms.Compose(test_transform_list)

    if aug:
        train_transform_list = [
            transforms.RandomResizedCrop(img_shape),
            # transforms.RandomCrop(img_shape, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(
                brightness=0.4,
                contrast=0.4,
                saturation=0.4,
                hue=0.2),
            transforms.ToTensor(),
            normalize,
        ]
        if flatten:
            train_transform_list.append(transforms.Lambda(lambda x: torch.flatten(x)))
    else:
        train_transform_list = list(test_transform_list)
    train_transform = transforms.Compose(train_transform_list)

    train_data = datasets.ImageFolder(traindir, train_transform)
    valid_data = datasets.ImageFolder(validdir, test_transform)

    train_queue = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers)

    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
    else:
        train_sampler = None

    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, shuffle=(train_sampler is None),
        num_workers=num_workers, pin_memory=True, sampler=train_sampler, drop_last=True)

    valid_queue = torch.utils.data.DataLoader(valid_data, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=num_workers)
    return train_queue, valid_queue, None
