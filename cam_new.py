from PIL import Image
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from torchvision import models, transforms
from torch.autograd import Variable
from torch.nn import functional as F
from torch import topk
import numpy as np
import cv2
import skimage.transform

image = Image.open("tractor.jpeg")
normalize = transforms.Normalize(
   mean=[0.485, 0.456, 0.406],
   std=[0.229, 0.224, 0.225]
)

# Preprocessing - scale to 224x224 for model, convert to tensor, 
# and normalize to -1..1 with mean/std for ImageNet

preprocess = transforms.Compose([
   transforms.Resize((224,224)),
   transforms.ToTensor(),
   normalize
])

display_transform = transforms.Compose([
   transforms.Resize((224,224))])
tensor = preprocess(image)

prediction_var = Variable((tensor.unsqueeze(0)), requires_grad=True)
model = models.resnet101(pretrained=True)
# checkpoint = torch.load('resnet_epoch_199.pth', map_location=torch.device('cpu')) 
# num = model.fc.in_features
# model.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(num, 200))
# model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
class SaveFeatures():
    features=None
    def __init__(self, m): self.hook = m.register_forward_hook(self.hook_fn)
    def hook_fn(self, module, input, output): self.features = ((output.cpu()).data).numpy()
    def remove(self): self.hook.remove()

final_layer = model._modules.get('layer4')
activated_features = SaveFeatures(final_layer)
prediction = model(prediction_var)
pred_probabilities = F.softmax(prediction).data.squeeze()
activated_features.remove()

def getCAM(feature_conv, weight_fc, class_idx):
    _, nc, h, w = feature_conv.shape
    cam = weight_fc[class_idx].dot(feature_conv.reshape((nc, h*w)))
    cam = cam.reshape(h, w)
    cam = cam - np.min(cam)
    cam_img = cam / np.max(cam)
    return [cam_img]

weight_softmax_params = list(model._modules.get('fc').parameters())
weight_softmax = np.squeeze(weight_softmax_params[0].cpu().data.numpy())

class_idx = topk(pred_probabilities,1)[1].int()
overlay = getCAM(activated_features.features, weight_softmax, class_idx)
imshow(display_transform(image))
imshow(skimage.transform.resize(overlay[0], tensor.shape[1:3]), alpha=0.5, cmap='jet')
# ol = skimage.transform.resize(overlay[0], tensor.shape[1:3])
# colormap = np.repeat(ol[:, :, np.newaxis], 3, axis=2)
# result = colormap + np.array(display_transform(image)) * 0.5
plt.axis('off')
# cv2.imwrite('CAM.jpg', result)
plt.margins(0,0)
plt.savefig('tractor_cam.png')