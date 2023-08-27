import os
import torch.utils.data as data
import numpy as np
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as tfs
from torchvision.transforms import functional as FF
import os,sys
import random
from PIL import Image
from torchvision.utils import make_grid

#from RandomMask1 import *
random.seed(2)
np.random.seed(2)


class RainDS_Dataset(data.Dataset):
    def __init__(self,path,train,crop=False,size=240,format='.png',dataset_type='all'):
        super(RainDS_Dataset,self).__init__()
        self.size=size
        # print('crop size',size)
        self.train=train
        self.crop = crop
        self.format=format
        
        dir_tmp = 'train' if self.train else 'test'
        
        self.gt_path = os.path.join(path,dir_tmp,'gt')
        
        self.gt_list = []
        self.rain_list = []
        
        raindrop_path = os.path.join(path,dir_tmp,'raindrop')
        rainstreak_path = os.path.join(path,dir_tmp,'rainstreak')
        streak_drop_path = os.path.join(path,dir_tmp,'rainstreak_raindrop')
        
        raindrop_names = os.listdir(raindrop_path)
        rainstreak_names = os.listdir(rainstreak_path)
        streak_drop_names = os.listdir(streak_drop_path)
        
        rd_input = []
        rd_gt = []
        
        rs_input = []
        rs_gt = []
        
        rd_rs_input=[]
        rd_rs_gt = []
        
        for name in raindrop_names:
            rd_input.append(os.path.join(raindrop_path,name))
            gt_name = name.replace('rd','norain')
            rd_gt.append(os.path.join(self.gt_path,gt_name))
            
        for name in rainstreak_names:
            rs_input.append(os.path.join(rainstreak_path,name))
            gt_name = name.replace('rain','norain')
            rs_gt.append(os.path.join(self.gt_path,gt_name))
            
        for name in streak_drop_names:
            rd_rs_input.append(os.path.join(streak_drop_path,name))
            gt_name = name.replace('rd-rain','norain')
            rd_rs_gt.append(os.path.join(self.gt_path,gt_name))
        
        
        if dataset_type=='all':
            self.gt_list += rd_gt
            self.rain_list += rd_input
            self.gt_list += rs_gt
            self.rain_list += rs_input
            self.gt_list += rd_rs_gt
            self.rain_list += rd_rs_input
        elif dataset_type=='rs':
            self.gt_list += rs_gt
            self.rain_list += rs_input           
        elif dataset_type=='rd':
            self.gt_list += rd_gt
            self.rain_list += rd_input
        elif dataset_type=='rsrd':
            self.gt_list += rd_rs_gt
            self.rain_list += rd_rs_input
                      
    def __getitem__(self, index):
        rain=Image.open(self.rain_list[index])
        clear_path = self.gt_list[index]
        clear=Image.open(clear_path)
        name = self.rain_list[index].split('/')[-1].split(".")[0]
        if not isinstance(self.size,str) and self.crop: 
            i,j,h,w=tfs.RandomCrop.get_params(clear,output_size=(self.size,self.size))
            clear=FF.crop(clear,i,j,h,w)
            rain = FF.crop(rain,i,j,h,w)
        
        if self.train:
            rain,clear =self.augData(rain.convert("RGB") ,clear.convert("RGB"))
        else:
            rain=tfs.ToTensor()(rain.convert("RGB"))
            clear=tfs.ToTensor()(clear.convert("RGB"))           
        return rain,clear,name
    def augData(self,data,target):
        rand_hor=random.randint(0,1)
        rand_rot=random.randint(0,3)
        data=tfs.RandomHorizontalFlip(rand_hor)(data)
        target=tfs.RandomHorizontalFlip(rand_hor)(target)
        if rand_rot:
            data=FF.rotate(data,90*rand_rot)
            target=FF.rotate(target,90*rand_rot)
        data=tfs.ToTensor()(data)
        target=tfs.ToTensor()(target)
        return data,target
    def __len__(self):
        return len(self.rain_list)

class AGAN_Dataset(data.Dataset):
    def __init__(self,path,train=False,crop=False,size=256,format='.png'):
        super(AGAN_Dataset,self).__init__()
        self.size=size
        self.InpaintSize = 64
        self.crop = crop
        # print('crop size',size)
        self.train=train
        self.format=format
        self.haze_imgs_dir=os.listdir(os.path.join(path,'data'))
        print('======>total number for training:',len(self.haze_imgs_dir))
        self.haze_imgs=[os.path.join(path,'data',img) for img in self.haze_imgs_dir]
        self.clear_dir=os.path.join(path,'gt')
    def __getitem__(self, index):
        haze=Image.open(self.haze_imgs[index])
        self.format = self.haze_imgs[index].split('/')[-1].split(".")[-1]
        while haze.size[0]<self.size or haze.size[1]<self.size :
            if isinstance(self.size,int):
                index=random.randint(0,10000)
                haze=Image.open(self.haze_imgs[index])
        img=self.haze_imgs[index]
        id=img.split('/')[-1].split("_")[0]
        clear_name=id+'_clean'+'.'+self.format
        clear=Image.open(os.path.join(self.clear_dir,clear_name))
        clear=tfs.CenterCrop(haze.size[::-1])(clear)
        if not isinstance(self.size,str) and self.crop:
            i,j,h,w=tfs.RandomCrop.get_params(haze,output_size=(self.size,self.size))
            haze=FF.crop(haze,i,j,h,w)
            clear=FF.crop(clear,i,j,h,w)
        haze,clear=self.augData(haze.convert("RGB") ,clear.convert("RGB"))
        return haze,clear,id
    def augData(self,data,target):
        if self.train:
            rand_hor=random.randint(0,1)
            rand_rot=random.randint(0,3)
            data=tfs.RandomHorizontalFlip(rand_hor)(data)
            target=tfs.RandomHorizontalFlip(rand_hor)(target)
            if rand_rot:
                data=FF.rotate(data,90*rand_rot)
                target=FF.rotate(target,90*rand_rot)

        data=tfs.ToTensor()(data)
        target=tfs.ToTensor()(target)
        return  data,target
    def __len__(self):
        return len(self.haze_imgs)

class Rain200_Dataset(data.Dataset):
    def __init__(self,path,train=False,crop=False,size=256,format='.tif',rand_inpaint=False,rand_augment=None):
        super(Rain200_Dataset,self).__init__()
        self.size=size
        self.rand_augment=rand_augment
        self.rand_inpaint=rand_inpaint
        self.InpaintSize = 64
        self.crop = crop
        # print('crop size',size)
        self.train=train
        self.format=format
        self.haze_imgs_dir=os.listdir(os.path.join(path,'rain','X2'))
        print('======>total number for training:',len(self.haze_imgs_dir))
        self.haze_imgs=[os.path.join(path,'rain','X2',img) for img in self.haze_imgs_dir]
        self.clear_dir=os.path.join(path,'norain')
    def __getitem__(self, index):
        haze=Image.open(self.haze_imgs[index])
        self.format = self.haze_imgs[index].split('/')[-1].split(".")[-1]
        while haze.size[0]<self.size or haze.size[1]<self.size :
            if isinstance(self.size,int):
                index=random.randint(0,10000)
                haze=Image.open(self.haze_imgs[index])
        img=self.haze_imgs[index]
        id=img.split('/')[-1].split(".")[0]
        clear_name=id[:-2]+'.'+self.format
        clear=Image.open(os.path.join(self.clear_dir,clear_name))
        clear=tfs.CenterCrop(haze.size[::-1])(clear)
        if not isinstance(self.size,str) and self.crop: 
            i,j,h,w=tfs.RandomCrop.get_params(haze,output_size=(self.size,self.size))
            haze=FF.crop(haze,i,j,h,w)
            clear=FF.crop(clear,i,j,h,w)
        haze,clear=self.augData(haze.convert("RGB") ,clear.convert("RGB"))
        return haze,clear,id
    def augData(self,data,target):
        if self.train:
            rand_hor=random.randint(0,1)
            rand_rot=random.randint(0,3)
            data=tfs.RandomHorizontalFlip(rand_hor)(data)
            target=tfs.RandomHorizontalFlip(rand_hor)(target)
            if rand_rot:
                data=FF.rotate(data,90*rand_rot)
                target=FF.rotate(target,90*rand_rot)

        data=tfs.ToTensor()(data)
        target=tfs.ToTensor()(target)
        
        return  data,target
    def __len__(self):
        return len(self.haze_imgs)

