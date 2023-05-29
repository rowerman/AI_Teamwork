# 库导入
import torch.nn.functional as F
import numpy as np
import torch 
from torch import nn
from PIL import Image
import matplotlib.pyplot as plt
import os
from torchvision import datasets, transforms,utils
from makeData import load_data
from visualize import visualize_Datasets
from model import Generator,Discriminator,weights_init,show_images,zyn
from torch.autograd import Variable
import time
import matplotlib.gridspec as gridspec
from torchvision.utils import save_image
import torchvision.utils as vutils


# ————————————————————————————————————————————————————————————————————————— #
##################################控制面板####################################

# 是否继续训练之前已有模型
if_countinue=1

# model name to be learn:
modelG_name="modelG/model_0_0.pt"
modelD_name="modelD/model_0_0.pt"

# 学习率
lr = 0.0002

# 生成器输入通道
noise_dim=100

# 优化器Adam用的beta1
beta1 = 0.5

# 训练轮次
num_epochs=100

image_size = 28

# 批大小
b_size =64

num_feature=3136

# gpu数目
ngpu = 1

##################################--------####################################
# —————————————————————————————————————————————————————————————————————————— #

zyn()

device = torch.device('cuda' if (torch.cuda.is_available() and ngpu > 0) else 'cpu')
print ("you are using "+str(device))
# Create the generator
# netG = Generator(ngpu)



# 第一次训练的时候请填1否则填0
datasets,dataloader=load_data(if_download=1,batch_size=b_size)

# 如果想看一眼数据集的话就取消下一行的注释
# visualize_Datasets(datasets)



f=open('txt', mode='r', buffering=-1, encoding=None, errors=None, newline=None, closefd=True, opener=None)
zyn=f.read()
print(zyn)


# 生成器&判别器の实例化&初始化：

netG=Generator(num_feature=num_feature,noise_dim=noise_dim,num_classes=10,device=device).to(device)
netD=Discriminator(device=device).to(device)

if if_countinue:
    netG.load_state_dict(torch.load(modelG_name))
    netD.load_state_dict(torch.load(modelD_name)) 

else:
    netG.apply(weights_init)
    netD.apply(weights_init)

# 优化器の初始化
optimizerD = torch.optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = torch.optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

criterion=nn.BCELoss()

# Label one-hot for G
label_1hots = torch.zeros(10,10,device=device)
for i in range(10):
    label_1hots[i,i] = 1
label_1hots = label_1hots.view(10,10,1,1)

# Label one-hot for D
label_fills = torch.zeros(10, 10, image_size, image_size,device=device)
ones = torch.ones(image_size, image_size,device=device)
for i in range(10):
    label_fills[i][i] = ones
label_fills = label_fills



img_list = []
G_losses = []
D_losses = []
D_x_list = []
D_z_list = []
loss_tep = 10
iters=0
# 开始训练
run_time=0
print("Starting Training Loop...")
# For each epoch

if __name__ == '__main__':
    for epoch in range(num_epochs):
        iters=1
        beg_time = time.time()
        # For each batch in the dataloader
        for i, data in enumerate(dataloader):
            
            ##############################################################################################
            #————————————————————————————————————————————————————————————————————————————————————————————#
            #————————————————————————————————————————判别器D的训练————————————————————————————————————————#
            #————————————————————————————————————————————————————————————————————————————————————————————#
            ##############################################################################################
            netD.zero_grad()
        
            # data拆包与batch制作
            # 图片值
            real_image = data[0] #这里输入图片为(64，1，28，28)的四维tensor(batch_size,channel_size,image_size,image_size)
            # 标签值
            Ten_real_label=data[1]
            
            
            # print(real_image.shape) # 输出(batch数量，维度，大小，大小) [64, 1, 28, 28]
 
            # 生成 lable 的 one-hot 向量，且设置对应类别位置是 1
            labels_onehot = np.zeros((real_image.shape[0], 10))
            labels_onehot[np.arange(real_image.shape[0]), Ten_real_label.numpy()] = 1

            real_image=real_image.to(device)
            Ten_real_label=Ten_real_label.to(device)
            # 真实数据标签和虚假数据标签，
            real_label = Variable(torch.from_numpy(labels_onehot).float()).to(device)  # 真实label对应类别是为1
            fake_label = Variable(torch.zeros(real_image.shape[0], 10,device=device)).to(device) # 假的label全是为0

            G_label = label_1hots[data[1]].to(device)
            D_label = label_fills[data[1]].to(device)

            # Forward pass real batch through D
            real_output = netD(real_image, D_label)
            real_output = real_output.view(-1)


            # 计算判别器的loss，我们希望它更接近正确答案，所以去和真实标签做交叉熵
            errD_real = criterion(real_output, real_label.view(-1))
            # 把这个loss反向传播回去
            errD_real.backward()
            
            # 计算一个判别器准确率AccuracyRate_D接下来显示出来
            similarity_D=torch.zeros(real_output.size(0),device=device)
            for i in range(real_output.size(0)):
                similarity_D[i]=(real_output[i]==(real_label.view(-1))[i])
            AccuracyRate_D = similarity_D.mean().item()


            # 生成随机向量，也就是噪声z，带有标签信息给生成器使用
            z = Variable(torch.randn(real_image.shape[0], noise_dim))
            z = np.concatenate((z.numpy(), labels_onehot), axis=1) # 噪音和one-hot 向量加在一起
            z = Variable(torch.from_numpy(z).float())
            
            # 用刚刚的随机噪音生成假图像fake
            fake_image = netG(z)
            
            # 用我们自己的判别器Discriminator去判别一下我们自己的Generator
            fake_output = netD(fake_image.detach(), D_label).view(-1)
            # 计算一下假的图像和全零的交叉熵，我们现在在训练D，所以我们希望判别器认为G生成的图片越假越好，和全零越贴合越好，这里于是算一下和全零的交叉熵
            errD_fake = criterion(fake_output, fake_label.view(-1))
            # 计算梯度反向传播
            errD_fake.backward()
            
            # 计算一个判别器准确率AccuracyRate_G接下来显示出来
            similarity_G=torch.zeros(fake_output.size(0),device=device)
            for i in range(fake_output.size(0)):
                similarity_G[i]=(fake_output[i]==(fake_label.view(-1))[i])
            AccuracyRate_G = similarity_G.mean().item()
            
            # 取出数据一会用
            D_G_z1 = fake_output.mean().item()
            
            # 把看见假数据和真数据的误差相加当作loss
            errD = errD_real + errD_fake
            # 更新D的optimizer
            optimizerD.step()





            ##############################################################################################
            #————————————————————————————————————————————————————————————————————————————————————————————#
            #————————————————————————————————————————生成器G的训练————————————————————————————————————————#
            #————————————————————————————————————————————————————————————————————————————————————————————#
            ##############################################################################################
            
            
            
            netG.zero_grad()
            # 用我们制作好的D去检测我们刚刚用G制作的假图片
            fake_output = netD(fake_image, D_label).view(-1)
            fake_output=fake_output.to(device)
            # 计算loss，因为我们在训练生成器G，所以我们希望这个生成器的结果越接近真实越好，这样说明我们的G足够优秀骗过了D
            #errG全部是相对值，所以变化会比较诡异
            errG = criterion(fake_output, real_label.view(-1))
            # 计算梯度反向传播
            errG.backward()
            D_G_z2 = fake_output.mean().item()
            # 更新D的optimizer
            optimizerG.step()



            ##############################################################################################
            #————————————————————————————————————————————————————————————————————————————————————————————#
            #———————————————————————————————————————————数据复盘——————————————————————————————————————————#
            #————————————————————————————————————————————————————————————————————————————————————————————#
            ##############################################################################################

            if(iters%25==0):
            # 实时显示训练数据
                end_time = time.time()
                run_t=run_time
                run_time = round(end_time-beg_time)
                run_time_of_this_iters=run_time-run_t
                print(
                    f'Epoch: [{epoch+1:0>{len(str(num_epochs))}}/{num_epochs}]',
                    f'iters: [{iters}/1093]',
                    f'Step: [{i+1:0>{len(str(len(dataloader)))}}/{len(dataloader)}]',
                    f'Loss-D: {errD.item():.4f}',
                    f'Loss-G: {errG.item():.4f}',
                    # f'D(x): {AccuracyRate_D:.4f}',
                    f'D(G(z)): [{D_G_z1:.4f}/{D_G_z2:.4f}]',
                    f'Time: {run_time}s',
                    f'Time_25_iter: {run_time_of_this_iters}s',
                    end='\n'
                )

            # 保存一下数据作图用
            G_losses.append(errG.item())
            D_losses.append(errD.item())
        
            # # 保存一下数据作图用
            # D_x_list.append(AccuracyRate_D)
            # D_z_list.append(D_G_z2)
        
            # 保存最佳网络结果
            if errG < loss_tep:
                torch.save(netG.state_dict(), "modelG/model_"+str(epoch+1)+"_"+str(iters)+".pt" )
                torch.save(netD.state_dict(), "modelD/model_"+str(epoch+1)+"_"+str(iters)+".pt" )
                torch.save(netG.state_dict(), "modelG/model_"+'0'+"_"+'0'+".pt" )
                torch.save(netD.state_dict(), "modelD/model_"+'0'+"_"+'0'+".pt" )
                loss_tep = errG
            if(iters%250==0):
                i = vutils.make_grid(fake_image, padding=2, normalize=True)
                fig = plt.figure(figsize=(4, 4))
                plt.imshow(np.transpose(i.to('cpu'), (1, 2, 0)))
                plt.axis('off')  # 关闭坐标轴
                plt.savefig("pictures_output/%d_%d.png" % (epoch+1, iters))
                plt.close(fig)
                f = open("errG/errG_"+str(epoch)+"_"+str(iters), "a")
                for i in G_losses:         
                    f.write(str(i)+" ")
                f.close()
                f = open("errD/errD_"+str(epoch)+"_"+str(iters), "a")
                for i in D_losses:         
                    f.write(str(i)+" ")
                f.close()
            iters+=1
        
        torch.save(netG.state_dict(), "modelG/model_"+str(epoch+1)+"_"+'0'+".pt" )
        torch.save(netD.state_dict(), "modelD/model_"+str(epoch+1)+"_"+'0'+".pt" )
        torch.save(netG.state_dict(), "modelG/model_"+'0'+"_"+'0'+".pt" )
        torch.save(netD.state_dict(), "modelD/model_"+'0'+"_"+'0'+".pt" )
        print()
        
        


