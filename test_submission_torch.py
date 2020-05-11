import sys
import pathlib
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as transforms
from model import Net
import torchvision
from tiny_loader import *
from torch.utils.data import Dataset
import torch.nn as nn

from resnet import resnet50, resnet101
def main():
    # Load the classes
    data_dir = pathlib.Path('./data/tiny-imagenet-200/train/')
    CLASSES = sorted([item.name for item in data_dir.glob('*')])
    im_height, im_width = 64, 64

    ckpt = torch.load('final_50/resnet_epoch_99.pth')
    model = resnet50(pretrained=False)
    num = model.fc.in_features
    model.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(num, 200))
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    
    magenet_mean, imagenet_std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

    data_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(imagenet_mean, imagenet_std)])

    # Loop through the CSV file and make a prediction for each line
    with open('eval_classified.csv', 'w') as eval_output_file:  # Open the evaluation CSV file for writing
        for line in pathlib.Path(sys.argv[1]).open():  # Open the input CSV file for reading
            image_id, image_path, image_height, image_width, image_channels = line.strip().split(
                ',')  # Extract CSV info

            print(image_id, image_path, image_height, image_width, image_channels)
            with open(image_path, 'rb') as f:
                img = Image.open(f).convert('RGB')
            img = data_transforms(img)[None, :]
            outputs = model(img)
            _, predicted = outputs.max(1)

            # Write the prediction to the output file
            eval_output_file.write('{},{}\n'.format(image_id, CLASSES[predicted]))


if __name__ == '__main__':
    main()

