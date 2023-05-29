import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import utils, datasets, transforms
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def visualize_Datasets(dataset):
    
    imgs = {}
    for x, y in dataset:
        if y not in imgs:
            imgs[y] = []
        elif len(imgs[y])!=10:
            imgs[y].append(x)
        elif sum(len(imgs[key]) for key in imgs)==100:
            break
        else:
            continue
        
    imgs = sorted(imgs.items(), key=lambda x:x[0])
    imgs = [torch.stack(item[1], dim=0) for item in imgs]
    imgs = torch.cat(imgs, dim=0)

    plt.figure(figsize=(10,10))
    plt.title("Training Images")
    plt.axis('off')
    imgs = utils.make_grid(imgs, nrow=10)
    plt.imshow(imgs.permute(1, 2, 0)*0.5+0.5)
