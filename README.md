# Final Project for CS 182
## Vision: Training a robust image classifier
* Various approaches to prevent overfitting
* Dataset augmentation
* Domain randomization
## Training dataset: Tiny-ImageNet Dataset
* Note that the code was using CUDA 10.1 and a Tesla V100 GPU. If there is no CUDA device available, default CPU option will be used, albeit slow. 
* Requirements: Install all the dependencies via 
  ```pip3 install -r requirements.txt```
* Start Training: 
  ```python train_resnet.py```
