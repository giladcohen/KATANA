from typing import Any, Tuple
import torch
from torchvision.datasets import VisionDataset
import numpy as np
from time import time

class TTADataset(VisionDataset):

    def __init__(self, data, y_gt, tta_size, *args, **kwargs) -> None:
        root = None
        super().__init__(root, *args, **kwargs)
        self.data = data
        self.y_gt = y_gt
        self.tta_size = tta_size
        assert type(self.data) == type(self.y_gt) == torch.Tensor, \
            'types of data, y_gt must be tensor type'
        self.img_shape = tuple(self.data.size()[1:])
        self.full_tta_size = (tta_size, ) + self.img_shape

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img, y_gt = self.data[index], self.y_gt[index]

        # first, duplicate the image to TTAs:
        img_ttas = np.nan * torch.ones(self.full_tta_size)

        # start = time()
        # now, transforming each image separately
        if self.transform is not None:
            for k in range(self.tta_size):
                img_ttas[k] = self.transform(img)
        # print('TTA transforms generation time: {}'.format(time() - start))

        return img_ttas, y_gt
