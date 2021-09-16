from .resnet import ResNet18, ResNet34, ResNet50, ResNet101

def get_strides(dataset: str):
    if dataset in ['cifar10', 'cifar100', 'svhn']:
        strides = [1, 2, 2, 2]
    elif dataset == 'tiny_imagenet':
        strides = [2, 2, 2, 2]
    else:
        raise AssertionError('Unsupported dataset {}'.format(dataset))
    return strides

def get_conv1_params(dataset: str):
    if dataset in ['cifar10', 'cifar100', 'svhn']:
        conv1 = {'kernel_size': 3, 'stride': 1, 'padding': 1}
    elif dataset == 'tiny_imagenet':
        conv1 = {'kernel_size': 7, 'stride': 1, 'padding': 3}
    else:
        raise AssertionError('Unsupported dataset {}'.format(dataset))
    return conv1

def get_model(moder_str):
    if moder_str == 'resnet18':
        return ResNet18
    elif moder_str == 'resnet34':
        return ResNet34
    elif moder_str == 'resnet50':
        return ResNet50
    elif moder_str == 'resnet101':
        return ResNet101
    else:
        raise AssertionError("network {} is unknown".format(moder_str))
