import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp
from torchvision.models.vgg import vgg19,vgg16
import torch.nn as nn




class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss,self).__init__()
        self.L1 = nn.L1Loss()
        self.mse = nn.MSELoss()
        vgg = vgg19(pretrained=True).eval()
        self.loss_net1 = nn.Sequential(*list(vgg.features)[:1]).eval() 
        self.loss_net3 = nn.Sequential(*list(vgg.features)[:3]).eval()
        self.loss_net5 = nn.Sequential(*list(vgg.features)[:5]).eval()
        self.loss_net9 = nn.Sequential(*list(vgg.features)[:9]).eval()
        self.loss_net13 = nn.Sequential(*list(vgg.features)[:13]).eval()
    def forward(self,x,y):
        loss1 = self.L1(self.loss_net1(x),self.loss_net1(y))
        loss3 = self.L1(self.loss_net3(x),self.loss_net3(y))
        loss5 = self.L1(self.loss_net5(x),self.loss_net5(y))
        loss9 = self.L1(self.loss_net9(x),self.loss_net9(y))
        loss13 = self.L1(self.loss_net13(x),self.loss_net13(y))
        #print(self.loss_net13(x).shape)
        loss = 0.2*loss1 + 0.2*loss3 + 0.2*loss5 + 0.2*loss9 + 0.2*loss13
        return loss



class PerceptualLoss2(nn.Module):
    def __init__(self):
        super(PerceptualLoss2,self).__init__()
        self.L1 = nn.L1Loss()
        self.mse = nn.MSELoss()
        vgg = vgg19(pretrained=True).eval()
        self.loss_net1 = nn.Sequential(*list(vgg.features)[:1]).eval()
        self.loss_net3 = nn.Sequential(*list(vgg.features)[:3]).eval()
    def forward(self,x,y):
        loss1 = self.L1(self.loss_net1(x),self.loss_net1(y))
        loss3 = self.L1(self.loss_net3(x),self.loss_net3(y))
        loss = 0.5*loss1+0.5*loss3
        return loss