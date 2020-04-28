import torch
import numpy as np
import torchvision.transforms as transforms
import torchvision
import torch.nn as nn
import torch.nn.functional as F

class BaseBlock(nn.Module):
    expand = 1
    def __init__(self, inC, outC, stride=1, dim_change=None):
        super(BaseBlock, self).__init__()
        self.conv1 = nn.Conv2d(inC, outC, stride=stride, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(outC)
        self.conv2 = nn.Conv2d(outC, outC, stride=1, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(outC)
        self.dim_change = dim_change
    def forward(self, x):
        res = x
        output = F.relu(self.bn1(self.conv1(x)))
        output = self.bn2(self.conv2(output))
        if self.dim_change:
            res = self.dim_change(res)
        output += res
        output = F.relu(output)
        return output

class BottleNeck(nn.Module):
    expand = 4
    def __init__(self, inC, outC, stride=1, dim_change=None):
        super(BottleNeck, self).__init__()
        self.conv1 = nn.Conv2d(inC, outC, stride=1, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(outC)
        self.conv2 = nn.Conv2d(outC, outC, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(outC)
        self.conv3 = nn.Conv2d(outC, outC * self.expand, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(outC * self.expand)
        self.dim_change = dim_change
    def forward(self, x):
        res = x
        output = F.relu(self.bn1(self.conv1(x)))
        output = F.relu(self.bn2(self.conv2(output)))
        output = self.bn3(self.conv3(output))
        if self.dim_change:
            res = self.dim_change(res)
        output += res
        output = F.relu(output)
        return output

class ResNet(nn.Module):
    def __init__(self, block, num_layers, classes=200):
        super(ResNet, self).__init__()
        self.inC = 64
        # ResNet-50
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.layer1 = self._layer(block, 64, num_layers[0], stride=1)
        self.layer2 = self._layer(block, 128, num_layers[1], stride=2)
        self.layer3 = self._layer(block, 256, num_layers[2], stride=2)
        self.layer4 = self._layer(block, 512, num_layers[3], stride=2)
        self.avgPool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expand, classes)

    def _layer(self, block, planes, num_layers, stride=1):
        dim_change = None
        if stride != 1 or planes != self.inC * block.expand:
            dim_change = nn.Sequential(nn.Conv2d(self.inC, planes * block.expand, kernel_size=1, stride=stride),
                                       nn.BatchNorm2d(planes * block.expand))
        net_layers = []
        net_layers.append(block(self.inC, planes, stride=stride, dim_change=dim_change))
        self.inC = planes * block.expand
        for i in range(1, num_layers):
            net_layers.append(block(self.inC, planes))
            self.inC = planes * block.expand
        return nn.Sequential(*net_layers)
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgPool(x)
        # x = F.avg_pool2d(x, 4)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
