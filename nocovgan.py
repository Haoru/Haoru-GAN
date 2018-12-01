import torch
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms
from torch import optim
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec


def showing(images,count):
    images=images.detach().numpy()[0:16,:]
    images=255*(0.5*images+0.5)
    images = images.astype(np.uint8)
    grid_length=int(np.ceil(np.sqrt(images.shape[0])))
    plt.figure(figsize=(4,4))
    width = int(np.sqrt((images.shape[1])))
    gs = gridspec.GridSpec(grid_length,grid_length,wspace=0,hspace=0)
    # gs.update(wspace=0, hspace=0)
    print('starting...')
    for i, img in enumerate(images):
        ax = plt.subplot(gs[i])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(img.reshape([width,width]),cmap = plt.cm.gray)
        plt.axis('off')
        plt.tight_layout()
    print('showing...')
    plt.tight_layout()
    plt.savefig('./GAN_Image/%d.png'%count, bbox_inches='tight')

def loadMNIST(batch_size):  #MNIST图片的大小是28*28
    trans_img=transforms.Compose([transforms.ToTensor()])
    trainset=MNIST('./data',train=True,transform=trans_img,download=True)
    testset=MNIST('./data',train=False,transform=trans_img,download=True)
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    trainloader=DataLoader(trainset,batch_size=batch_size,shuffle=True,num_workers=10)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=10)
    return trainset,testset,trainloader,testloader

batch_size=128
num_epoch=100
z_dimension=100


class discriminator(nn.Module):
    def __init__(self):
        super(discriminator,self).__init__()
        self.dis=nn.Sequential(
            nn.Linear(784,300),
            nn.LeakyReLU(0.2),
            nn.Linear(300,150),
            nn.LeakyReLU(0.2),
            nn.Linear(150,1),
            nn.Sigmoid()

        )
    def forward(self,x):
        x=self.dis(x)
        return x
    
class generator(nn.Module):
    def __init__(self):
        return super(generator,self).__init__()
        self.gen=nn.Sequential(
            nn.Linear(100,150),
            nn.ReLU(True),
            nn.Linear(150,300),
            nn.ReLU(True),
            nn.Linear(300,784),
            nn.Tanh()

        )
    def forward(self, x):
        x=self.gen(x)
        return x
if __name__=="__name__":
    D=discriminator()
    G=generator(z_dimension)
    criterion=nn.BCELoss()# Binary Cross Entropy 二分类的交叉损失商
    num_img=100
    d_optimizer=torch.optim.Adam(D.parameters(),lr=0.0003)
    g_optimizer=torch.optim.Adam(D.parameters(),lr=0.0003)

    count=0
    epoch=100
    gepoch=1
    for i in range(epoch):
        for (img,label) in loadMNIST.trainloader:
            real_img=img.view(num_img,-1)
            real_label=torch.ones(num_img)
            fake_label=torch.zeros(num_img)

        
            real_out=D(real_img)
            d_loss_real=criterion(real_img,real_label)
            real_scores=real_out

            z=torch.randn(num_img,z_dimension)
            fake_img=G(z)
            fake_out=D(fake_img)
            d_loss_fake=criterion(fake_out,fake_label)
            fake_scores=fake_out

            d_loss=d_loss_real+d_loss_fake
            d_optimizer.zero_grad()
            d_loss.backword()
            d_optimizer.step()

            for j in range(gepoch):
                fake_label=torch.ones(num_img)
                z=torch.randn(num_img,z_dimension)
                fake_img=G(z)
                output=D(fake_img)
                g_loss=criterion(output,fake_label)
                g_optimizer.zero_grad()
                g_loss.backward()
                g_optimizer.step()
        print('Epoch [{}/{}],d_loss:{:.6f},g_loss:{:.6f}''D real: {:.6f}, D fake: {:.6f}'.format(
            i,epoch,d_loss.data[0],g_loss.data[0],real_scores.data.mean(),
            fake_scores.data.mean()
            ))
        showing(fake_img,count)
        plt.show()
        count+=1

