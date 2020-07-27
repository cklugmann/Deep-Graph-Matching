import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import ImageGrid
import torch


def cuda_to_numpy(x):
    return x.detach().cpu().numpy()


def renormalized_tensor(img_tensor, img_mean=[0.485, 0.456, 0.406], img_std=[0.229, 0.224, 0.225]):
    """
    Undo the normalization transformation for an image tensor.
    :param img_tensor: tensor of shape (N,3,H,W)
    :param img_mean: list of mean pixel values per channel
    :param img_std: list of std of pixel values per channel
    :return: np.array of integer values between 0 and 255 of shape (N,H,W,3)
    """
    img = np.moveaxis(cuda_to_numpy(img_tensor), [0, 1, 2, 3], [0, 3, 1, 2])
    img_mean = np.array(img_mean)
    img_std = np.array(img_std)
    img = img_std.reshape(1, 1, 1, -1) * img + img_mean.reshape(1, 1, 1, -1)
    img *= 255
    return img.astype(int)


def show_images_and_keypoints(img_s, img_t, points_s, points_t, mask, pred_t, file=None):
    """
    Visualizes source and target images with their respective keypoints
    :param img_s: image batch of shape (N,3,H,W)
    :param img_t: image batch of shape (N,3,H,W)
    :param points_s: source keypoints, tensor of shape (N,num_points,2)
    :param points_t: target keypoints, tensor of shape (N,num_points,2)
    :param mask: Binary mask for restoring original nodes, tensor of shape (N,num_points,1)
    :param pred_t: predicted target keypoints, tensor of shape (N,num_points,2)
    :param file: path, where image gets written to
    """

    img1, img2 = renormalized_tensor(img_s), renormalized_tensor(img_t)
    points1, points2 = cuda_to_numpy(points_s), cuda_to_numpy(points_t)
    num_nodes = cuda_to_numpy(torch.sum(mask, dim=1).squeeze())
    points_pred = cuda_to_numpy(pred_t)
    num_images = img1.shape[0]
    gridsize = (2, num_images)

    fig = plt.figure()
    grid = ImageGrid(fig, 111, nrows_ncols=gridsize, axes_pad=0.1)

    for idx, ax in enumerate(grid):
        ax.axis('off')
        bid = idx % num_images
        n = num_nodes[bid]
        colors = cm.rainbow(np.linspace(0, 1, n))
        if idx < num_images:
            points = points1[bid, :n]
            X, Y = points[:, 0], points[:, 1]
            ax.scatter(X, Y, s=1, color=colors)
            ax.imshow(img1[bid])
        else:
            points = points2[bid, :n]
            points_p = points_pred[bid, :n]
            X, Y = points[:, 0], points[:, 1]
            Xpred, Ypred = points_p[:, 0], points_p[:, 1]
            ax.scatter(X, Y, s=1, color=colors)
            ax.scatter(Xpred, Ypred, s=1, color=colors, marker='x')
            ax.imshow(img2[bid])

    if file is not None:
        plt.savefig(file)
    else:
        plt.show()