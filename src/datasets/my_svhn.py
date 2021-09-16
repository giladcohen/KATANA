from torchvision.datasets import SVHN
from torchvision.datasets.utils import verify_str_arg
import os
import numpy as np
from PIL import Image
import torch

class MySVHN(SVHN):

    def __init__(self, *args, **kwargs) -> None:
        train = kwargs['train']
        download = kwargs['download']
        if train:
            split = 'train'
        else:
            split = 'test'
        super(MySVHN, self).__init__(kwargs['root'], split, transform=kwargs['transform'],
                                     download=download)  # just for the transform and download

        # import here rather than at top of file because this is
        # an optional dependency for torchvision
        import scipy.io as sio

        # reading(loading) mat file as array
        loaded_mat = sio.loadmat(os.path.join(self.root, self.filename))

        self.data = loaded_mat['X']
        # loading from the .mat file gives an np array of type np.uint8
        # converting to np.int64, so that we have a LongTensor after
        # the conversion from the numpy array
        # the squeeze is needed to obtain a 1D tensor
        self.targets = loaded_mat['y'].astype(np.int64).squeeze()

        # the svhn dataset assigns the class label "10" to the digit 0
        # this makes it inconsistent with several loss functions
        # which expect the class labels to be in the range [0, C-1]
        np.place(self.targets, self.targets == 10, 0)
        self.data = np.transpose(self.data, (3, 0, 1, 2))

        # limit dataset from 73257/26032 train/test samples to 72000/26000 train/test samples
        if train:
            self.data = self.data[:72000]
            self.targets = self.targets[:72000]
        else:
            self.data = self.data[:26000]
            self.targets = self.targets[:26000]

        self.classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # img = Image.fromarray(np.transpose(img, (1, 2, 0)))
        if type(img) != torch.Tensor:
            img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target
