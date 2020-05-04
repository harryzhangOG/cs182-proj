from PIL import Image
from torchvision import models, transforms
import torch.nn as nn
from torch.nn import functional as F
import torch
import numpy as np
import cv2
from resnet import resnet101
import torchvision
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from sklearn.manifold import TSNE
from tiny_loader import *

checkpoint = torch.load('resnet_epoch_199.pth', map_location=torch.device('cpu'))


net = resnet101(pretrained=False)
num = net.fc.in_features
net.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(num, 200))
net.load_state_dict(checkpoint['model_state_dict'])
net.eval()
features = []

def hook_feature(module, input, output):
    features.append(output)

net._modules.get('avgpool').register_forward_hook(hook_feature)

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
])

trainloader = iter(train_imagenet_loader(False))
valloader = iter(val_imagenet_loader(False))
points = [[] for i in range(200)]
for X_train, y_train in valloader:
    # X_train, y_train = next(trainloader)
    features = []
    net(X_train)
    feat = features[0].squeeze()
    feat = feat.detach().cpu().numpy()
    feat_proj = TSNE(n_components=2).fit_transform(feat)
    print(feat_proj.shape)
    # points = [[] for i in range(200)]
    idx = y_train.detach().cpu().numpy()
    for i in range(128):
        if feat_proj.shape[0] == 8 and i >= 8:
            break
        points[idx[i]].append(feat_proj[i])
points = [np.array(p).T for p in points]
plt.figure(num=None)
plt.xlim(-10, 10)
plt.ylim(-20, 20)
for p in points:
    if p.size == 0:
        continue
    plt.scatter(p[0], p[1], s=5.5)
plt.savefig('tsne.jpg')