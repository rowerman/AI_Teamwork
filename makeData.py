import numpy as np
import torch 
from torch import nn
from PIL import Image
import matplotlib.pyplot as plt
import os
from torchvision import datasets, transforms,utils


def load_data(if_download=False,batch_size=64):
    transform = transforms.Compose([transforms.ToTensor(),
                               transforms.Normalize(mean=[0.5],std=[0.5])])
    print('*******************************************')
    if if_download:
        print('*****       downloading data...       *****')
        
    train_data = datasets.MNIST(root = "./data/",
                            transform=transform,
                            train = True,
                            download = if_download)
    print('*****    downloading data comleted    *****')
    test_data = datasets.MNIST(root="./data/",
                           transform = transform,
                           train = False)

    dataset=train_data+test_data
    
    loader = torch.utils.data.DataLoader(dataset,
                                           batch_size=batch_size,
                                          shuffle=True,
                                          num_workers=2)

    
    print(f'***** Total Size Of Datasets is:{len(dataset)} *****')
    print('*******************************************')
    
    return dataset,loader

