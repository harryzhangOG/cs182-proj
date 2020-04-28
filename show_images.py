import numpy as np
import matplotlib.pyplot as plt

def show_images_horizontally(images, labels=[], un_normalize=False, fig_size=(15, 7)):
    """Show images in jupyter notebook horizontally w/ labels as title.
    Parameters
    ----------
    images: pytorch Tensor of shape (#images, #channels, height, width)
    labels: pytorch Tensor of shape (#images, label)
        labels should be encoded in number like 0, 1 .. and so on.
    un_normalize: bool
        indicate whether to perform unnormalization operation for rendering.
    fig_size: tuple
    """

    fig = plt.figure(figsize=fig_size)
    num_imgs = images.shape[0]
    for i in range(num_imgs):
        fig.add_subplot(1, num_imgs, i + 1)

        # render image tensor
        img = images[i]
        npimg = img.numpy()
        if un_normalize:
            npimg = npimg / 2 + 0.5
        npimg = np.transpose(npimg, (1, 2, 0))

        # generate label as title
        if labels:
            plt.title(lookup_label[labels[i][0]])
        plt.imshow(npimg, cmap='Greys_r')
        plt.axis('off')
        plt.show()
