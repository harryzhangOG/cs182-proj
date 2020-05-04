import io
import requests
from PIL import Image
import torch
from torchvision import models, transforms
from torch.autograd import Variable
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import cv2
import pdb
from resnet import *
# Reference: https://github.com/metalbubble/CAM/
checkpoint = torch.load('resnet_epoch_199.pth', map_location=torch.device('cpu')) 

net = resnet101(pretrained=True)
# num = net.fc.in_features
# net.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(num, 200))
# net.load_state_dict(checkpoint['model_state_dict'])
net.eval()
# hook the feature extractor
features_blobs = []
def hook_feature(module, input, output):
    features_blobs.append(output.data.cpu().numpy())

net.cpu()
net._modules.get('layer4').register_forward_hook(hook_feature)
# get the softmax weight
params = list(net.parameters())
weight_softmax = np.squeeze(params[-2].data.numpy())


def returnCAM(feature_conv, weight_softmax, class_idx):
    # generate the class activation maps upsample to 256x256
    size_upsample = (256, 256)
    bz, nc, h, w = feature_conv.shape
    output_cam = []
    for idx in class_idx:
        cam = weight_softmax[idx].dot(feature_conv.reshape((nc, h*w)))
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / 1e-5+(np.max(cam))
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(cv2.resize(cam_img, size_upsample))
    return output_cam

normalize = transforms.Normalize(
   mean=[0.485, 0.456, 0.406],
   std=[0.229, 0.224, 0.225]
)
preprocess = transforms.Compose([
   transforms.Resize((32, 32)),
   transforms.ToTensor(),
   normalize
])
img_name = 'bug.JPEG'
img_pil = Image.open(img_name)
# img_tensor = preprocess(img_pil)
# img_variable = Variable(img_tensor.unsqueeze(0))
img_tensor = preprocess(img_pil).unsqueeze(0)
logit = net(img_tensor)
h_x = F.softmax(logit, dim=1).data.squeeze()
probs, idx = h_x.sort(0, True)
probs = probs.numpy()
print(idx)
idx = idx.numpy()
# classes1 = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
# classes2 = ['up', 'left', 'down', 'right']
# if rotate:
#     classes = classes2
# else:
#     classes = classes1
# # output the prediction
# for i in range(0, len(classes)):
#     print('{:.3f} -> {}'.format(probs[i], classes[idx[i]]))
# # generate class activation mapping for the top1 prediction
CAMs = returnCAM(features_blobs[0], weight_softmax, [idx[0]])
print(CAMs[0].max())

# render the CAM and output
print('output CAM.jpg for the top1 prediction')
img = cv2.imread(img_name)
height, width, _ = img.shape
heatmap = cv2.applyColorMap(cv2.resize(CAMs[0],(width, height)), cv2.COLORMAP_JET)
result = heatmap * 0.5 + img * 0.5
cv2.imwrite('CAM.jpg', result)
