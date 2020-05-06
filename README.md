# Final Project for CS 182
## Vision: Training a robust image classifier
* Various approaches to prevent overfitting
* Dataset augmentation
* Domain randomization
## Training dataset: Tiny-ImageNet Dataset
* Tiny ImageNet dataset is a subset of ImageNet. Instead of 100 classes, Tiny ImageNet only has 200 classes. There are about 100k images in the dataset.

## Prerequisites
* Note that the code was using CUDA 10.1 and a Tesla V100 GPU. If there is no CUDA device available, default CPU option will be used, albeit slow. 
* Requirements: Install all the dependencies via 
  ```pip3 install -r requirements.txt```
* Start Training: 
  ```python train_resnet.py```

## Dataset Preparation
* The format of the original validation dataset is different than the training dataset. To format the validation dataset into the correct way, please run:
```python val_format.py```
This script extracts all the subfolders in the images folder and take them out of the subdirectories so that each subdirectory now is treated as a class, and can be passed into ImageFolder processor provided by PyTorch.
* The dataset extraction step is essentially done in `tiny_loader.py`. The gist is to make use of PyTorch's built-in `ImageFolder`, which takes a directory of directories and treat each sub-directory as the label of images in it.
  
## Training Details
* The classifier's backbone is a ResNet-101 architecture. I compared my implementation with PyTorch's official implementation, and there wasn't much difference.
* After 200 epochs, I modified the last FC layer of the network and added Dropout. Then I train for another 200 epochs.
* I also wrote a learning rate scheduler. The learning rate of the network decays by a factor of 0.5 each 10 epochs in the first 200 epochs, and then decays by a factor of 0.7 each 20 epochs in the second 200 epochs.
* The training script also saves the model periodically:
  * I save the model very 10 epochs. The code that achieves that is:
  ```python
  if save_e % 10 == 0:
      print('saving models for epoch ' + str(save_e))
      torch.save({'epoch': epoch,
                  'model_state_dict': net50.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict()
                 }, 'resnet_epoch_' + str(epoch + 1) + '.pth')
      print('model saved')
   ```
* I also save the training accuracy, training loss, validation accuracy, and validation loss in 4 NumPy arrays. They can be used later to plot the training process. You can optionally set up a TensorBoard to monitor the training process in real time, although more difficult. 

## Visualization and Interpretability
* The project has several built-in methods to visualize and interpret the results of classification:
  * Class activation map (CAM): CAM is a covenient choice because batch normalizations are built in ResNet.
  * t-SNE: t-SNE makes visualization of the projected features. I used the off-the-shelf SKLEARN implementation and it is very slow. One run of t-SNE on the full training dataset takes about 50 minutes, and one run of t-SNE on the full validation dataset takes about 4 minutes. 
  * Filters visualization: In the early stages of the network, it is possible to visualize the learned weights in the network. I wrote a script to extract the convolutional filters in the first layer, and visualize the filters as RGB images. 
  
## Generate classification csv file 
* Generate predictions of images listed in ``` eval.csv ``` to ```eval_classified.csv``` by running 
``` python 
python test_submission_torch.py eval.csv 
```


