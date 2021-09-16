import torch
import numpy as np
import os
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.model_selection import train_test_split

from .my_cifar10 import MyCIFAR10
from .my_cifar100 import MyCIFAR100
from .my_svhn import MySVHN
from .tiny_imagenet import TinyImageNet
from src.utils import get_image_shape

BASE_DATASET_DIR = '/tmp/datasets'
def dataset_factory(dataset):
    if dataset == 'cifar10':
        data_dir = os.path.join(BASE_DATASET_DIR, 'cifar10')
        database = MyCIFAR10
    elif dataset == 'cifar100':
        data_dir = os.path.join(BASE_DATASET_DIR, 'cifar100')
        database = MyCIFAR100
    elif dataset == 'svhn':
        data_dir = os.path.join(BASE_DATASET_DIR, 'svhn')
        database = MySVHN
    elif dataset == 'tiny_imagenet':
        data_dir = os.path.join(BASE_DATASET_DIR, 'tiny_imagenet')
        database = TinyImageNet
    else:
        raise AssertionError('dataset {} is not supported'.format(dataset))

    img_size = get_image_shape(dataset)[0]
    pad_size = int(img_size / 8)

    if dataset in ['cifar10', 'cifar100', 'tiny_imagenet']:
        train_transform = transforms.Compose([
            transforms.RandomCrop(img_size, padding=pad_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
    else:
        train_transform = transforms.Compose([
            transforms.RandomCrop(img_size, padding=pad_size),
            transforms.ToTensor(),
        ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    return data_dir, database, train_transform, test_transform


def get_train_valid_loader(dataset,
                           batch_size,
                           rand_gen,
                           valid_size=0.1,
                           shuffle=True,
                           num_workers=4,
                           pin_memory=False):
    """
    Utility function for loading and returning train and valid
    multi-process iterators over the CIFAR-10 train_dataset. A sample
    9x9 grid of the images can be optionally displayed.
    If using CUDA, num_workers should be set to 1 and pin_memory to True.
    Params
    ------
    - dataset: name of the dataset.
    - batch_size: how many samples per batch to load.
    - augment: whether to apply the data augmentation scheme
      mentioned in the paper. Only applied on the train split.
    - rand_gen: rand_gen for reproducibility.
    - valid_size: percentage split of the training set used for
      the validation set. Should be a float in the range [0, 1].
    - shuffle: whether to shuffle the train/validation indices.
    - num_workers: number of subprocesses to use when loading the train_dataset.
    - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
      True if using GPU.
    Returns
    -------
    - train_loader: training set iterator.
    - valid_loader: validation set iterator.
    """
    error_msg = "[!] valid_size should be in the range [0, 1]."
    assert ((valid_size >= 0) and (valid_size <= 1)), error_msg

    data_dir, database, train_transform, test_transform = dataset_factory(dataset)

    # load the dataset
    train_dataset = database(
        root=data_dir, train=True,
        download=True, transform=train_transform
    )

    valid_dataset = database(
        root=data_dir, train=True,
        download=True, transform=test_transform
    )

    num_train_val = len(train_dataset)
    num_val       = int(np.floor(valid_size * num_train_val))
    indices = list(range(num_train_val))

    train_idx, val_idx = \
        train_test_split(indices, test_size=num_val, random_state=rand_gen, shuffle=shuffle, stratify=train_dataset.targets)
    train_idx.sort()
    val_idx.sort()

    train_dataset.data = train_dataset.data[train_idx]
    train_dataset.targets = np.asarray(train_dataset.targets)[train_idx]
    valid_dataset.data = valid_dataset.data[val_idx]
    valid_dataset.targets = np.asarray(valid_dataset.targets)[val_idx]

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    return train_loader, valid_loader, train_idx, val_idx

def get_test_loader(dataset,
                    batch_size,
                    num_workers=4,
                    pin_memory=False,
                    transforms=None):
    """
    Utility function for loading and returning a multi-process
    test iterator over the CIFAR-10 dataset.
    If using CUDA, num_workers should be set to 1 and pin_memory to True.
    Params
    ------
    - dataset: name of the dataset.
    - batch_size: how many samples per batch to load.
    - num_workers: number of subprocesses to use when loading the dataset.
    - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
      True if using GPU.
    Returns
    -------
    - data_loader: test set iterator.
    """

    data_dir, database, _, test_transform = dataset_factory(dataset)

    if transforms is None:
        transforms = test_transform

    dataset = database(
        root=data_dir, train=False,
        download=True, transform=transforms
    )

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory, drop_last=False
    )

    return data_loader

def get_normalized_tensor(loader: torch.utils.data.DataLoader, img_shape, batch_size=None):
    """ Returning a normalized tensor"""
    if batch_size is None:
        batch_size = loader.batch_size
    size = len(loader.dataset)
    X = -1.0 * np.ones(shape=(size, img_shape[2], img_shape[0], img_shape[1]), dtype=np.float32)
    for batch_idx, (inputs, targets) in enumerate(loader):
        b = batch_idx * batch_size
        e = b + targets.shape[0]
        X[b:e] = inputs.cpu().numpy()

    return X
