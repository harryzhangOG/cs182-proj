import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tiny_dataset import TinyImageNet
from torch.autograd import Variable
from show_images import show_images_horizontally
from resnet import BottleNeck, BaseBlock, ResNet

def train():
    normalize = transforms.Normalize((.5, .5, .5), (.5, .5, .5))

    augmentation = transforms.RandomApply([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomAffine(10, translate=(0.1, 0.2), scale=(0.8, 1.1), shear=3),
        transforms.RandomResizedCrop(64)], p=.9)

    training_transform = transforms.Compose([
        transforms.Lambda(lambda x: x.convert("RGB")),
        augmentation,
        transforms.ToTensor(),
        normalize])

    valid_transform = transforms.Compose([
        transforms.Lambda(lambda x: x.convert("RGB")),
        transforms.ToTensor(),
        normalize])

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

    resnet50 = ResNet(BottleNeck, [3, 4, 6, 3], 200)
    resnet50.to(device)

    cost = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(resnet50.parameters(), 1e-3)

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
            prediction = resnet50(data)
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
            prediction = resnet50(data)
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






if __name__ == "__main__":
    train()

