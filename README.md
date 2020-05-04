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

## Visualization and Interpretability
* The project has several built-in methods to visualize and interpret the results of classification:
  * Class activation map (CAM): CAM is a covenient choice because batch normalizations are built in ResNet.
  * t-SNE: t-SNE makes visualization of the projected features. I used the off-the-shelf SKLEARN implementation and it is very slow. One run of t-SNE on the full training dataset takes about 50 minutes, and one run of t-SNE on the full validation dataset takes about 4 minutes. 
  * Filters visualization: In the early stages of the network, it is possible to visualize the learned weights in the network. I wrote a script to extract the convolutional filters in the first layer, and visualize the filters as RGB images. 
  
