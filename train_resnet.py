import torch
import numbers
import numpy as np
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tiny_dataset import TinyImageNet
from torch.autograd import Variable
from show_images import show_images_horizontally
from resnet import resnet50
from PIL import ImageFilter
from PIL import Image
import cv2

# Helper class for adding Gaussian noises to images
class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, image):
        a = np.array(image)

        noisy_image = np.zeros(a.shape, np.float32)
        gaussian0 = np.random.normal(0, 10, (64, 64))
        noisy_image[:, :, 0] = a[:, :, 0] + gaussian0
        gaussian1 = np.random.normal(0, 10, (64, 64))
        noisy_image[:, :, 1] = a[:, :, 1] + gaussian1
        gaussian2 = np.random.normal(0, 10, (64, 64))
        noisy_image[:, :, 2] = a[:, :, 2] + gaussian2

        cv2.normalize(noisy_image, noisy_image, 0, 255, cv2.NORM_MINMAX, dtype=-1)
        noisy_image = noisy_image.astype(np.uint8)

        return Image.fromarray(noisy_image)
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

# Helper class for adding Gaussian filter (blur) to images
class AddGaussianFilter(object):
    def __init__(self, radius):
        if isinstance(radius, numbers.Number):
            self.min_radius = radius
            self.max_radius = radius
        elif isinstance(radius, list):
            if len(radius) != 2:
                raise Exception(
                    "`radius` should be a number or a list of two numbers")
            if radius[1] < radius[0]:
                raise Exception(
                    "radius[0] should be <= radius[1]")
            self.min_radius = radius[0]
            self.max_radius = radius[1]
        else:
            raise Exception(
                "`radius` should be a number or a list of two numbers")

    def __call__(self, image):
        radius = np.random.uniform(self.min_radius, self.max_radius)
        return image.filter(ImageFilter.GaussianBlur(radius)) 


def train():
    # Normalize the data. Not needed if we have batchnorm
    normalize = transforms.Normalize((.5, .5, .5), (.5, .5, .5))

    """
    Data Augmentation:
        Randomly apply the following actions to images in each epoch:
            1. Random Horizontal Flip
            2. Random Vertical Flip
            3. Random Affine transformation
            4. Random crop
            5. Add Gaussian noise
            6. Add Blurring
    """
    augmentation = transforms.RandomApply([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomAffine(10, translate=(0.1, 0.2), scale=(0.8, 1.1), shear=3),
        transforms.RandomResizedCrop(64),
        AddGaussianNoise(0., 1.),
        AddGaussianFilter([0, 1])], p=.7)

    training_transform = transforms.Compose([
        transforms.Lambda(lambda x: x.convert("RGB")),
        augmentation,
        transforms.ToTensor(),
        normalize])

    valid_transform = transforms.Compose([
        transforms.Lambda(lambda x: x.convert("RGB")),
        transforms.ToTensor(),
        normalize])

    # EXTRACT THE DATASET FROM OUR CUSTOM DATASET
    root = 'data/tiny-imagenet-200'
    training_set = TinyImageNet(root, 'train', transform=training_transform, in_memory=False)
    training_set = torch.utils.data.DataLoader(training_set, batch_size=128)
    valid_set = TinyImageNet(root, 'val', transform=valid_transform, in_memory=False)
    valid_set = torch.utils.data.DataLoader(valid_set, batch_size=128)
    # tmpiter = iter(DataLoader(training_set, batch_size=10, shuffle=True))
    # for _ in range(5):
    #     images, labels = tmpiter.next()
    #     show_images_horizontally(images, un_normalize=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using ", device)

    net50 = resnet50()
    net50.to(device)

    cost = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net50.parameters(), 1e-3)

    trainLoss = []
    valLoss = []
    trainAcc = []
    valAcc = []
    total_loss = []

    for epoch in range(20):
        # Training loss
        total_loss = []
        for i, batch in enumerate(training_set, 0):
            data, output = batch
            data, output = data.to(device), output.to(device)
            prediction = net50(data)
            loss = cost(prediction, output)
            closs = loss.item()
            total_loss.append(closs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i % 100 == 0:
                trainLoss.append(closs)
                print('[Epoch %d, Iteration %d] Training Loss: %.5f' % (epoch+1, i, closs))
                closs = 0
        # Validation loss
        for i, batch in enumerate(valid_set, 0):
            data, output = batch
            data, output = data.to(device), output.to(device)
            prediction = net50(data)
            loss = cost(prediction, output)
            vloss = loss.item()
            if i % 100 == 0:
                valLoss.append(vloss)
                print('[Epoch %d, Iteration %d] Validation Loss: %.5f' % (epoch+1, i, vloss))
                vloss = 0
        
        # Calculating Accuracy
        correctHits = 0
        total=0


        for batches in trainset:
            data,output = batches
            data,output = data.to(device),output.to(device)
            prediction = net50(data)
            _,prediction = torch.max(prediction.data,1)  #returns max as well as its index
            total += output.size(0)
            correctHits += (prediction==output).sum().item()
        
        trainAcc.append(correctHits / total)
        print('Training accuracy on epoch ',epoch+1,'= ',str((correctHits/total)*100))


        correctHits = 0
        total=0


        for batches in valid_set:
            data,output = batches
            data,output = data.to(device),output.to(device)
            prediction = net50(data)
            _,prediction = torch.max(prediction.data,1)  #returns max as well as its index
            total += output.size(0)
            correctHits += (prediction==output).sum().item()
        
        valAcc.append(correctHits / total)
        print('Validation accuracy on epoch ',epoch+1,'= ',str((correctHits/total)*100))

        print('saving models')
        torch.save({
                'epoch': epoch,
                'model_state_dict': net50.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, 'resnetlight_rotate_epoch' + str(epoch + 1) + '.pth')
        print('model saved')

    return trainLoss, valLoss, trainAcc, valAcc, total_loss






if __name__ == "__main__":
    trainLoss, valLoss, trainAcc, valAcc, total_loss = train()
    np.save('trainLoss.npy', trainLoss)
    np.save('valLoss.npy', valLoss)
    np.save('trainAcc.npy', trainAcc)
    np.save('valAcc.npy', valAcc)

