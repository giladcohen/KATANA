import numpy as np
import os

rand_gen = np.random.RandomState(123456)
dataset_dirs = [
    '/tmp/results/cifar10',
    '/tmp/results/cifar100',
    '/tmp/results/svhn',
    '/tmp/results/tiny_imagenet'
]

for dataset_dir in dataset_dirs:
    os.makedirs(dataset_dir, exist_ok=True)
    if 'cifar' in dataset_dir or 'tiny_imagenet' in dataset_dir:
        test_test_size, test_size = 2500, 10000
    else:
        test_test_size, test_size = 2500, 26000

    test_test_inds = rand_gen.choice(np.arange(test_size), test_test_size, replace=False)
    test_test_inds.sort()
    test_val_inds = np.asarray([i for i in np.arange(test_size) if i not in test_test_inds])

    if os.path.exists(os.path.join(dataset_dir, 'test_test_inds.npy')):  # protection from overwriting test indices
        print('test indices were already selected for {}'.format(dataset_dir))
    else:
        np.save(os.path.join(dataset_dir, 'test_test_inds.npy'), test_test_inds)
        np.save(os.path.join(dataset_dir, 'test_val_inds.npy'), test_val_inds)

print('done')
