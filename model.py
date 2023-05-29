import torch.nn.functional as F
import numpy as np
import torch 
from torch import nn
from PIL import Image
import matplotlib.pyplot as plt
import os
from torchvision import datasets, transforms,utils
import matplotlib.gridspec as gridspec
from torchvision.utils import save_image

# 弃用的用于识别的cnn
# class CNN(nn.Module):
#     def __init__(self):
#         super(CNN,self).__init__()
#         self.embedding=nn.Linear()
#         self.conv1 = nn.Conv2d(1,32,kernel_size=3,stride=1,padding=1)
#         self.pool = nn.MaxPool2d(2,2)
#         self.conv2 = nn.Conv2d(32,64,kernel_size=3,stride=1,padding=1)
#         self.fc1 = nn.Linear(64*7*7,1024)#两个池化，所以是7*7而不是14*14
#         self.fc2 = nn.Linear(1024,512)
#         self.fc3 = nn.Linear(512,10)
# #         self.dp = nn.Dropout(p=0.5)
#     def forward(self,x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))

#         x = x.view(-1, 64 * 7* 7)#将数据平整为一维的 
#         x = F.relu(self.fc1(x))
# #         x = self.fc3(x)
# #         self.dp(x)
#         x = F.relu(self.fc2(x))   
#         x = self.fc3(x)  
# #         x = F.log_softmax(x,dim=1) NLLLoss()才需要，交叉熵不需要
#         return x
    
    


def make_CNN(channels,max_pool,kernel_size,stride,padding,active,):
    net = []
    for i in range(len(channels)-1):
        net.append(nn.Conv2d(in_channels=channels[i], out_channels=channels[i+1],
                                 kernel_size=kernel_size[i], padding=padding[i],stride=stride[i], bias=False))#这里kernel_size
        if i == 0:
            net.append(nn.LeakyReLU(0.2))
        elif active[i] == "LR":
            net.append(nn.BatchNorm2d(num_features=channels[i+1]))
            net.append(nn.LeakyReLU(0.2,inplace=True))
        elif active[i] == "sigmoid":
            net.append(nn.Sigmoid())
        elif active[i] == "tanh":
            net.append(nn.Tanh())
        if max_pool[i]:
            net.append(nn.MaxPool2d((2, 2)))
            
    return nn.Sequential(*net) #组装在一起

class Generator(nn.Module):
    def __init__(self, num_feature=3136,noise_dim=100, num_classes=10,device='cpu'):
        super(Generator, self).__init__()
        self.ngpu=1
        self.noise_dim = noise_dim
        self.num_classes = num_classes
        self.device=device
        channels=[1,50,250,50,25,1]
        max_pool=[0,0,0,0,0]
        active = ["LR", "LR", "LR", "LR", "tanh"]
        stride = [1, 1, 1,1,2]
        padding=[1,1,1,1,0]
        kernel_size=[3,3,3,3,2]
        self.generator=make_CNN(channels,max_pool,kernel_size,stride,padding,active).to(device)
        self.fc = nn.Linear(noise_dim+num_classes, num_feature).to(device)

    def forward(self, x):
        x=x.to(self.device)
        x = self.fc(x)
        x = x.view(x.size(0), 1, 56, 56)
        x = self.generator(x)

        return x  # [batch_size, 1, 28, 28]
    
    
    
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class Discriminator(nn.Module):  # 输入图片批为[128, 1, 28, 28]
    def __init__(self,device='cpu'):
        super(Discriminator, self).__init__()
        self.ngpu=1
        self.device=device
        channels= [32,64,128, 256, 512,1024]
        max_pool=[0,0,0,1,1]
        kernel_size=[5,5,5,5,5]
        padding=[2,2,2,2,2]
        stride=[1,1,1,1,1]
        active = ["LR", "LR", "LR", "LR", "sigmoid"]
        
        self.discriminator=make_CNN(channels,max_pool,kernel_size,stride,padding,active)

        self.image = nn.Sequential(
            # input is (nc) x 32 x 32
            nn.Conv2d(1, 16, 4, 2, 1, bias=False),#这里第三个数
            nn.LeakyReLU(0.2, inplace=True)
            # state size. (ndf) x 16 x 16
        )
        self.label = nn.Sequential(

            nn.Conv2d(10, 16, 4, 2, 1, bias=False),#这里第三个数
            nn.LeakyReLU(0.2, inplace=True)

        )
        self.fc = nn.Sequential(
            nn.Linear(3* 3 * 1024, 2048),  # 全连接层
            nn.LeakyReLU(0.2, True),
            nn.Dropout(0.02),
            nn.Linear(2048, 512),
            nn.LeakyReLU(0.2, True),
            nn.Dropout(0.02),
            nn.Linear(512, 10),  # 分类
            nn.Sigmoid()
        )
 
    def forward(self, x,label):  # x: [batch_size, 1, 28, 28]

        label=self.label(label)
        x=self.image(x).to(self.device)

        data = torch.cat(tensors=(x, label), dim=1).to(self.device)
        out = self.discriminator(data).to(self.device)

        out = out.view(out.size(0), -1)
        out=self.fc(out)
        return out  # [batch_size, 10]
    

# 定义展示图片的函数
def show_images(images):  # 定义画图工具
    print('images: ', images.shape)
    images = np.reshape(images, [images.shape[0], -1])
    sqrtn = int(np.ceil(np.sqrt(images.shape[0])))
    sqrtimg = int(np.ceil(np.sqrt(images.shape[1])))
 
    fig = plt.figure(figsize=(sqrtn, sqrtn))
    gs = gridspec.GridSpec(sqrtn, sqrtn)
    gs.update(wspace=0.05, hspace=0.05)
 
    for i, img in enumerate(images):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(img.reshape([sqrtimg, sqrtimg]))
    return


def zyn():
    f=open('txt', mode='r', buffering=-1, encoding=None, errors=None, newline=None, closefd=True, opener=None)
    zyn=f.read()
    print(zyn)