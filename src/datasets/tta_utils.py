import torch
import numpy as np
import PIL
import torchvision.transforms as transforms
from time import time
import logging

import src.datasets.my_transforms as my_transforms
from src.utils import get_image_shape, pytorch_evaluate
from src.datasets.tta_dataset import TTADataset

def get_tta_transforms(dataset: str, gaussian_std: float=0.005, soft=False, clip_inputs=False):
    img_shape = get_image_shape(dataset)
    n_pixels = img_shape[0]

    if clip_inputs:
        clip_min, clip_max = 0.0, 1.0
    else:
        clip_min, clip_max = -np.inf, np.inf

    if dataset in ['cifar10', 'cifar100', 'tiny_imagenet']:
        p_hflip = 0.5
    else:
        p_hflip = 0.0

    tta_transforms = transforms.Compose([
        my_transforms.Clip(0.0, 1.0),  # To fix a bug where an ADV image has minus small value, applying gamma yields Nan
        my_transforms.ColorJitterPro(
            brightness=[0.8, 1.2] if soft else [0.6, 1.4],
            contrast=[0.85, 1.15] if soft else [0.7, 1.3],
            saturation=[0.75, 1.25] if soft else [0.5, 1.5],
            hue=[-0.03, 0.03] if soft else [-0.06, 0.06],
            gamma=[0.85, 1.15] if soft else [0.7, 1.3]
        ),
        transforms.Pad(padding=int(n_pixels / 2), padding_mode='edge'),  # padding to double the image size for rotation
        transforms.RandomAffine(
            degrees=[-8, 8] if soft else [-15, 15],
            translate=(1/16, 1/16),
            scale=(0.95, 1.05) if soft else (0.9, 1.1),
            shear=None,
            resample=PIL.Image.BILINEAR,
            fillcolor=None
        ),
        transforms.GaussianBlur(kernel_size=5, sigma=[0.001, 0.25] if soft else [0.001, 0.5]),
        transforms.CenterCrop(size=n_pixels),
        transforms.RandomHorizontalFlip(p=p_hflip),
        my_transforms.GaussianNoise(0, gaussian_std),
        my_transforms.Clip(clip_min, clip_max)
    ])
    return tta_transforms

def get_tta_logits(dataset, args, net, X, y, tta_size, num_classes):
    """ Calculating the TTA output logits out of the inputs, transforms, to the tta_dir"""
    tta_transforms = get_tta_transforms(dataset, args.gaussian_std, args.soft_transforms, args.clip_inputs)
    tta_dataset = TTADataset(
        torch.from_numpy(X),
        torch.from_numpy(y),
        tta_size,
        transform=tta_transforms)
    tta_loader = torch.utils.data.DataLoader(
        tta_dataset, batch_size=1, shuffle=False,
        num_workers=args.num_workers, pin_memory=torch.cuda.is_available())

    logger = logging.getLogger()
    start = time()
    with torch.no_grad():
        tta_logits = pytorch_evaluate(net, tta_loader, ['logits'],
                                      (-1,) + tta_dataset.img_shape, {'logits': (-1, tta_size, num_classes)},
                                      verbose=True)[0]
    logger.info('Finished running DNN inference to fetch all the TTA logits. It took {} seconds'.format(time() - start))

    return tta_logits
