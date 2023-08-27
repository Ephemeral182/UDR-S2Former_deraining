import argparse
import os
from os.path import exists, join as join_paths
import torch
import numpy as np
import warnings
from tqdm import tqdm

from torchvision.utils import save_image,make_grid
from torch.utils.data import DataLoader

from dataloader_udr import *
from metrics import *
from psnr_ssim import *
from UDR_S2Former import *

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--cuda', action='store_true', help='use GPU computation')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--tile', type=int, default=320, help='Tile size, None for no tile during testing (testing as a whole)')
parser.add_argument('--tile_overlap', type=int, default=64, help='Overlapping of different tiles')
parser.add_argument('--scale', type=int, default=1, help='scale factor: 1, 2, 3, 4, 8') 
parser.add_argument('--dataset_type', type=str, default='raindrop_real', help='raindrop_syn/raindrop_real/rain200h/rain200l/agan ') 
parser.add_argument('--dataset_raindrop_syn', type=str, default='/home/csx/data/RainDS/RainDS_syn/', help='path of syn dataset')
parser.add_argument('--dataset_raindrop_real', type=str, default='/home/csx/data/RainDS/RainDS_real/', help='path of real dataset')  
parser.add_argument('--dataset_rain200h', type=str, default='/data/SJ/00YeTian/rain_100/rain_data_test_Heavy/rain_heavy_test', help='path of Rain200h dataset')
parser.add_argument('--dataset_rain200l', type=str, default='/data/SJ/00YeTian/rain_100/rain_data_test_light/rain_light_test', help='path of Rain200l dataset')  
parser.add_argument('--dataset_agan', type=str, default='/home/csx/data/AGAN-datat/test_a', help='path of agan dataset')
parser.add_argument('--savepath', type=str, default='./out/', help='path of output image') 
parser.add_argument('--model_path', type=str, default='pretrained/udrs2former_', help='path of SnowFormer checkpoint') 


opt = parser.parse_args()

if opt.dataset_type == 'raindrop_syn':
    snow_test = DataLoader(dataset=RainDS_Dataset(opt.dataset_raindrop_syn,train=False,dataset_type='rsrd'),batch_size=1,shuffle=False,num_workers=4)
elif opt.dataset_type == 'raindrop_real':
    snow_test = DataLoader(dataset=RainDS_Dataset(opt.dataset_raindrop_real,train=False,dataset_type='rsrd'),batch_size=1,shuffle=False,num_workers=4)
elif opt.dataset_type == 'rain200h':
    snow_test = DataLoader(dataset=Rain200_Dataset(opt.dataset_rain200h,train=False),batch_size=1,shuffle=False,num_workers=4)
elif opt.dataset_type == 'rain200l':
    snow_test = DataLoader(dataset=Rain200_Dataset(opt.dataset_rain200l,train=False),batch_size=1,shuffle=False,num_workers=4)
elif opt.dataset_type == 'agan':
    snow_test = DataLoader(dataset=AGAN_Dataset(opt.dataset_agan,train=False),batch_size=1,shuffle=False,num_workers=4)


netG_1 = Transformer(img_size=(opt.tile,opt.tile)).cuda()

if __name__ == '__main__':   

    ssims = []
    psnrs = []
    rmses = []
    opt.model_path = opt.model_path + opt.dataset_type + '.pth'
    g1ckpt1 = opt.model_path
    ckpt = torch.load(g1ckpt1)
    netG_1.load_state_dict(ckpt)

    savepath_dataset = os.path.join(opt.savepath,opt.dataset_type)
    if not os.path.exists(savepath_dataset):
        os.makedirs(savepath_dataset)
    loop = tqdm(enumerate(snow_test),total=len(snow_test))

    for idx,(haze,clean,name) in loop:
        
        with torch.no_grad():
                
                haze = haze.cuda();clean = clean.cuda()

                b, c, h, w = haze.size()

                tile = min(opt.tile, h, w)
                print(tile)
                tile_overlap = opt.tile_overlap
                sf = opt.scale

                stride = tile - tile_overlap
                h_idx_list = list(range(0, h-tile, stride)) + [h-tile]
                w_idx_list = list(range(0, w-tile, stride)) + [w-tile]
                E1 = torch.zeros(b, c, h*sf, w*sf).type_as(haze)
                W1 = torch.zeros_like(E1)
                E2 = torch.zeros(b, c, h*sf, w*sf).type_as(haze)
                W2 = torch.zeros_like(E2)

                for h_idx in h_idx_list:
                    for w_idx in w_idx_list:
                        in_patch = haze[..., h_idx:h_idx+tile, w_idx:w_idx+tile]
                        out_patch1,_ = netG_1(in_patch)
                        out_patch1 = out_patch1[0]
                        out_patch_mask1 = torch.ones_like(out_patch1)
                        E1[..., h_idx*sf:(h_idx+tile)*sf, w_idx*sf:(w_idx+tile)*sf].add_(out_patch1)
                        W1[..., h_idx*sf:(h_idx+tile)*sf, w_idx*sf:(w_idx+tile)*sf].add_(out_patch_mask1)
                dehaze = E1.div_(W1)

                save_image(dehaze,os.path.join(savepath_dataset,'%s.png'%(name)),normalize=False)


                ssim1=calculate_ssim(dehaze*255,clean*255,crop_border=0,test_y_channel=True)
                psnr1=calculate_psnr(dehaze*255,clean*255,crop_border=0,test_y_channel=True)

                ssims.append(ssim1)
                psnrs.append(psnr1)

                print('Generated images %04d of %04d' % (idx+1, len(snow_test)))
                print('ssim:',(ssim1))
                print('psnr:',(psnr1))

        ssim = np.mean(ssims)
        psnr = np.mean(psnrs)
        print('ssim_avg:',ssim)
        print('psnr_avg:',psnr)
 