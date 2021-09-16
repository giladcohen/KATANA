'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, activation=F.relu):
        super(BasicBlock, self).__init__()
        self.activation = activation
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = self.activation(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.activation(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, activation=F.relu):
        super(Bottleneck, self).__init__()
        self.activation = activation
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = self.activation(self.bn1(self.conv1(x)))
        out = self.activation(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = self.activation(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, activation='relu', conv1=None, strides=None):
        super(ResNet, self).__init__()
        self.in_planes = 64
        if activation == 'relu':
            self.activation = F.relu
        elif activation == 'softplus':
            self.activation = F.softplus
        else:
            raise AssertionError('activation function {} was not expected'.format(activation))

        self.conv1 = nn.Conv2d(3, 64, kernel_size=conv1['kernel_size'], stride=conv1['stride'],
                               padding=conv1['padding'], bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=strides[0])
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=strides[1])
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=strides[2])
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=strides[3])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, self.activation))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        net = {}
        out = self.activation(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        # with torch.no_grad():
        net['embeddings'] = out
        out = self.linear(out)
        net['logits'] = out
        net['probs'] = F.softmax(out, dim=1)
        net['preds'] = net['probs'].argmax(dim=1)
        return net


def ResNet18(num_classes, activation, conv1, strides):
    return ResNet(BasicBlock, [2,2,2,2], num_classes, activation, conv1, strides)

def ResNet34(num_classes, activation, conv1, strides):
    return ResNet(BasicBlock, [3,4,6,3], num_classes, activation, conv1, strides)

def ResNet50(num_classes, activation, conv1, strides):
    return ResNet(Bottleneck, [3,4,6,3], num_classes, activation, conv1, strides)

def ResNet101(num_classes, activation, conv1, strides):
    return ResNet(Bottleneck, [3,4,23,3], num_classes, activation, conv1, strides)

def ResNet152(num_classes, activation, conv1, strides):
    return ResNet(Bottleneck, [3,8,36,3], num_classes, activation, conv1, strides)

def test():
    net = ResNet18()
    y = net(torch.randn(1,3,32,32))
    print(y.size())
