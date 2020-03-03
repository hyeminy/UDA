import torch.nn as nn
import torch.utils.model_zoo as model_zoo

from trans_norm import TransNorm1d, TransNorm2d

import math


#transnorm_resnet.resnet50(pretrained=True)
__all__ = ['ResNet', 'resnet50']

model_urls = {
'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
}


def resnet50(pretrained=False, **network_config): # pretrained=True로 해서 호출 됨

    for key, value in network_config.items():
        print("{0} : {1}".format(key, value))


    model = ResNet(Bottleneck, [3,4,6,3], **network_config)

    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))

    return model


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, type=None):

        super(ResNet, self).__init__()

        self.inplanes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3 ,bias=False)
        if type == 'normal':
            self.bn1 = nn.BatchNorm2d(64)
        else:
            self.bn1 = TransNorm2d(64)

        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], type=type)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, type=type)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, type=type)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, type=type)

        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0]*m.kernel_size[1]*m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, type=None):

        downsample = None

        if stride !=1 or self.inplanes != planes *block.expansion:
            if type == 'normal':
                downsample = nn.Sequential(
                                            nn.Conv2d(self.inplanes, planes*block.expansion,
                                                      kernel_size = 1, stride=stride, bias = False),
                                            nn.BatchNorm2d(planes*block.expansion)
                )

            else:
                downsample = nn.Sequential(
                                            nn.Conv2d(self.inplanes, planes*block.expansion,
                                                      kernel_size=1, stride=stride, bias = False),
                                            TransNorm2d(planes * block.expansion)
                )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, type)) # type 추가하기
        self.inplanes = planes * block.expansion
        for i in range(1 , blocks):
            layers.append(block(self.inplanes, planes, type=type))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x





class Bottleneck(nn.Module):

    expansion = 4

    #def __init__(self, inplanes, planes, stride=1, downsample=None, type='selection'):
    def __init__(self, inplanes, planes, stride=1, downsample=None, type=None): # type이 어떻게 넘어가는지 확인 용
        super(Bottleneck, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)

        if type == 'normal':
            self.bn1 = nn.BatchNorm2d(planes)

        else:
            self.bn1 = TransNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)

        if type == 'normal':
            self.bn2 = nn.BatchNorm2d(planes)

        else:
            self.bn2 = TransNorm2d(planes)

        self.conv3 = nn.Conv2d(planes, planes*4, kernel_size=1, bias=False)

        if type == 'normal':
            self.bn3 = nn.BatchNorm2d(planes*4)
        else:
            self.bn3 = TransNorm2d(planes*4)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out