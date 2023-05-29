import torch
import numpy as np
from torch.autograd import Variable
import torch 
from torch import nn
from model import Generator
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import os

from model import zyn

########################################################
# by第八组:
# 
#     报告:张瀚文,許昊晟,郑燮宇
#     代码:张奕宁
#     接口文件:刘俊杰
# 
# 注：感谢許昊晟同学的台式机和台式机的2070对本项目的大力支持
########################################################


#       ->CO$$$O?>   -OOO$$$$?:       :?O$$OO7-     >OO-      ?O?   7OO$$$OC!          .!C$$$O>-            
#     7HMH?!:::!7?.  7MMH:::7NMH:  .CMM$!:::?NMQ!   $MM>     .HMH-  HMM?::>HMN>       >NM$:-:$MM7           
#    CMM?            >MM$    $MM! .$MM>      -HMN:  OMN!     .QMH-  QMM-   7MM$       >MMO-  OMN!           
#   .MMH   :$$$$$O.  >MMN$$QNN?-  :NMQ.       OMM?  OMM!     .QMH-  QMMC>>CMMQ-        -OMMMMM$-            
#    HMN-   --:HMM-  >MMQ-:?MMO   :NMH-      .$MM>  OMN!     .QMH-  QMM$CC?!.         >$N$>->QMNC           
#    :HMN>     QMM-  >MMQ   !MMH.  7MMQ-    -OMM?   7MM$.    ?MM?   QMM:              NMM-   :MMN           
#      >OQHQ$QQNQO.  >MM$    >NMQ   -CQHQ$$QHQ?-     :CQHQ$$HHO!    $MM!              -OHNQQQHQC.           

class AiGcMn():
    def __init__(self):    
        self.G=Generator()
        self.G.load_state_dict(torch.load("model/model_0_0.pt")) 

    def generator(self,Labels):
        
        # self.Labels = Labels
        labels_onehot = np.zeros((64, 10))
        labels_onehot[np.arange(64), Labels.numpy()] = 1
        z = Variable(torch.randn(64, 100))
        z = np.concatenate((z.numpy(), labels_onehot), axis=1) # 噪音和one-hot 向量加在一起
        z = Variable(torch.from_numpy(z).float())
        output=self.G(z)
        return output


# 示例代码：输出20张π的前64位

def EXAMPLE_PI_20():
    Label='3141592653589793238462643383279502884197169399375105820974944592'
    Labels=[]
    for c in Label:
        Labels.append(int(c))
    Labels=torch.tensor(Labels)
    filename = r'/home/tim/workspace/test.txt'
    

    if not(os.path.exists('EXAMPLE_PI_OUTPUT')):
        os.mkdir('EXAMPLE_PI_OUTPUT')
    
    print("图片生成中................................")
    for j in range(20):
        a=AiGcMn()
        output=a.generator(Labels)
        i = vutils.make_grid(output, padding=2, normalize=True)
        fig = plt.figure(figsize=(5, 5))
        plt.imshow(np.transpose(i.to('cpu'), (1, 2, 0)))
        plt.axis('off')  # 关闭坐标轴
        plt.savefig("EXAMPLE_PI_OUTPUT/test_pic_%d.png" % (j))
        plt.close(fig)


def EXAMPLE_20():
    Label=input("请输入要生成的数字(不超过64位的整数),如果直接回车默认输出π :   ")
    if(Label==""):
        EXAMPLE_PI_20()
        return
    Labels=np.zeros(64)
    i=0
    for c in Label:
        Labels[i]=int(c)
        i+=1
    Labels=torch.IntTensor(Labels)

    if not(os.path.exists('EXAMPLE_'+str(Label[:5])+'OUTPUT')):
        os.mkdir('EXAMPLE_'+str(Label[:5])+'OUTPUT')
    print("图片生成中..........................   (完毕后的乱码是彩蛋,请缩小视图)")
    for j in range(20):
        a=AiGcMn()
        output=a.generator(Labels)[:len(Label)]
        i = vutils.make_grid(output, padding=2, normalize=True)
        fig = plt.figure(figsize=(5, 5))
        plt.imshow(np.transpose(i.to('cpu'), (1, 2, 0)))
        plt.axis('off')  # 关闭坐标轴
        plt.savefig('EXAMPLE_'+str(Label[:5])+'OUTPUT'+"/test_pic_%d.png" % (j))
        plt.close(fig)

EXAMPLE_20()
zyn("txts/txt2")