import pytorch_lightning as pl
from pytorch_lightning import callbacks
from pytorch_lightning.accelerators import accelerator
from pytorch_lightning.core.hooks import CheckpointHooks
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import ModelCheckpoint,DeviceStatsMonitor,EarlyStopping,LearningRateMonitor
from pytorch_lightning.trainer.connectors import *
from pytorch_lightning.cli import LightningCLI
from pytorch_lightning import seed_everything

import torch
import torch.nn as nn
import torch.nn.functional as F

from dataloader_udr import *
from metrics import *
from psnr_ssim import *
from copy import deepcopy
import tensorboardX

from loss.CL1 import L1_Charbonnier_loss, PSNRLoss 
from loss.perceptual import PerceptualLoss2

from UDR_S2Former import *
#from restormer import *

#Set seed
seed = 42 #Global seed set to 42
seed_everything(seed)
from pytorch_lightning.loggers import WandbLogger,TensorBoardLogger
logger = TensorBoardLogger(r'tb_logs', name='udrs2former')
lr_monitor = LearningRateMonitor(logging_interval='step')

class EMA(nn.Module):
    """ Model Exponential Moving Average V2 from timm"""
    def __init__(self, model, decay=0.9999):
        super(EMA, self).__init__()
        # make a copy of the model for accumulating moving average of weights
        self.module = deepcopy(model)
        self.module.eval()
        self.decay = decay

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model):
        self._update(model, update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m)

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)

class CoolSystem(pl.LightningModule):
    
    def __init__(self, 
                 train_datasets,
                 train_bs,
                 test_datasets,
                 test_bs,
                 val_datasets,
                 val_bs,
                 dataset_type,
                 initlr,
                 weight_decay,
                 crop_size,
                 crop_size_test,
                 num_workers,
                 ):
        super(CoolSystem, self).__init__()

         # train/val/test datasets
        self.train_datasets = train_datasets
        self.train_batchsize = train_bs
        self.test_datasets = test_datasets
        self.test_batchsize = test_bs
        self.validation_datasets = val_datasets
        self.val_batchsize = val_bs
        self.dataset_type = dataset_type

        #Train setting
        self.initlr = initlr #initial learning
        self.weight_decay = weight_decay #optimizers weight decay
        self.crop_size = crop_size #random crop size
        self.crop_size_test = crop_size_test
        self.num_workers = num_workers

        #loss_function
        self.loss_f = PSNRLoss()
        self.loss_l1 = nn.L1Loss()
        self.loss_per = PerceptualLoss2()

        # model
        self.model = Transformer(img_size=(crop_size,crop_size))
        # self.save_hyperparameters()

    def forward(self, x):
        y = self.model(x)
        return y
    
    def configure_optimizers(self):
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.initlr,betas=[0.9,0.999])
        
        scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer,base_lr=self.initlr,max_lr=1.2*self.initlr,step_size_up=400,cycle_momentum=False)

        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        x, y, _ = batch

        # to calculate loss
        y1 = nn.functional.interpolate(y, scale_factor=0.5, mode='bicubic')
        y2 = nn.functional.interpolate(y, scale_factor=0.25, mode='bicubic')
        y3 = nn.functional.interpolate(y, scale_factor=0.125, mode='bicubic')
        # forward process
        y_list,var_list = self.forward(x)

        loss_f = self.loss_f(y,y_list[1]) + self.loss_f(y1,y_list[2]) + self.loss_f(y2,y_list[3]) + self.loss_f(y3,y_list[4])
        loss_f = 0.5*(loss_f/4.0) + self.loss_f(y,y_list[0])
        loss_per = self.loss_per(y,y_list[0])

        s = torch.exp(-var_list[0])
        sr_ = torch.mul(y_list[0] ,s)
        hr_ = torch.mul(y, s)
        loss_uncertarinty0 =  self.loss_l1(sr_,hr_) + 2* torch.mean(var_list[0])
        s1 = torch.exp(-var_list[1])
        sr_1 = torch.mul(y_list[1] ,s1)
        hr_1 = torch.mul(y, s1)
        loss_uncertarinty1 =  self.loss_l1(sr_1,hr_1) + 2* torch.mean(var_list[1])
        s2 = torch.exp(-var_list[2])
        sr_2 = torch.mul(y_list[2] ,s2)
        hr_2 = torch.mul(y1, s2)
        loss_uncertarinty2 =  self.loss_l1(sr_2,hr_2) + 2* torch.mean(var_list[2])
        s3 = torch.exp(-var_list[3])
        sr_3 = torch.mul(y_list[3] ,s3)
        hr_3 = torch.mul(y2, s3)
        loss_uncertarinty3 =  self.loss_l1(sr_3,hr_3) + 2* torch.mean(var_list[3])
        s4 = torch.exp(-var_list[4])
        sr_4 = torch.mul(y_list[4] ,s4)
        hr_4 = torch.mul(y3, s4)
        loss_uncertarinty4 =  self.loss_l1(sr_4,hr_4) + 2* torch.mean(var_list[4])
        loss_uncertarinty = (loss_uncertarinty0 + loss_uncertarinty1 + loss_uncertarinty2 + loss_uncertarinty3 + loss_uncertarinty4)/ 5.0
        loss = (loss_f + 0.2*loss_per + loss_uncertarinty)
        self.log('train_loss', loss)
        self.log('uncertainty_loss', loss_uncertarinty)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        x, y, _ = batch
        b, c, h, w = x.size()

        tile = min(self.crop_size, h, w)
        tile_overlap = 32
        sf = 1

        stride = tile - tile_overlap
        h_idx_list = list(range(0, h-tile, stride)) + [h-tile]
        w_idx_list = list(range(0, w-tile, stride)) + [w-tile]
        E1 = torch.zeros(b, c, h*sf, w*sf).type_as(x)
        W1 = torch.zeros_like(E1)
        E2 = torch.zeros(b, c, h*sf, w*sf).type_as(x)
        W2 = torch.zeros_like(E2)
        for h_idx in h_idx_list:
            for w_idx in w_idx_list:
                in_patch = x[..., h_idx:h_idx+tile, w_idx:w_idx+tile]
                out_patch1,_ = self.forward(in_patch)
                out_patch1 = out_patch1[0]
                out_patch_mask1 = torch.ones_like(out_patch1)
                E1[..., h_idx*sf:(h_idx+tile)*sf, w_idx*sf:(w_idx+tile)*sf].add_(out_patch1)
                W1[..., h_idx*sf:(h_idx+tile)*sf, w_idx*sf:(w_idx+tile)*sf].add_(out_patch_mask1)

        y_hat = E1.div_(W1)

        loss = self.loss_f(y,y_hat) + 0.2*self.loss_per(y,y_hat)
        psnr = PSNR(y_hat,y) # no test in y channel, just train
        ssim = SSIM(y_hat,y) # no test in y channel, just train

        self.log('val_loss', loss)
        self.log('psnr', psnr)
        self.log('ssim', ssim)
        
        self.trainer.checkpoint_callback.best_model_score

        return {'val_loss': loss, 'psnr': psnr,'ssim':ssim}


    def train_dataloader(self):
        if self.dataset_type == 'raindrop_syn':
            train_set = RainDS_Dataset(self.train_datasets,train=True,crop=True,size=self.crop_size)
        elif self.dataset_type == 'raindrop_real':
            train_set = RainDS_Dataset(self.train_datasets,train=True,crop=True,size=self.crop_size)
        elif self.dataset_type == 'agan':
            train_set = AGAN_Dataset(self.train_datasets,train=True,crop=True,size=self.crop_size)
        else:
            train_set = Rain200_Dataset(self.train_datasets,train=True,crop=True,size=self.crop_size)
        
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=self.train_batchsize, shuffle=True, num_workers=self.num_workers)

        return train_loader
    
    def val_dataloader(self):
        if self.dataset_type == 'raindrop_syn':
            val_set = RainDS_Dataset(self.train_datasets,train=False,crop=True,dataset_type='rsrd',size=self.crop_size_test)
        elif self.dataset_type == 'raindrop_real':
            val_set = RainDS_Dataset(self.train_datasets,train=False,crop=True,dataset_type='rsrd',size=self.crop_size_test)
        elif self.dataset_type == 'agan':
            val_set = AGAN_Dataset(self.train_datasets,train=False,crop=True,size=self.crop_size_test)
        else:
            val_set = Rain200_Dataset(self.train_datasets,train=False,crop=True,size=self.crop_size_test)
        val_loader = torch.utils.data.DataLoader(val_set, batch_size=self.val_batchsize, shuffle=False, num_workers=self.num_workers)
        
        return val_loader
    
def cli_main():
    checkpoint_callback = ModelCheckpoint(
    monitor='psnr',
    filename='RainDrop-Base-epoch{epoch:02d}-psnr{psnr:.3f}-ssim{ssim:.3f}',
    auto_insert_metric_name=False,   
    every_n_epochs=1,
    save_top_k=6,
    mode = "max",
    save_last=True
    )
    trainer_defaults = {'devices':[1,2,3],'callbacks':[checkpoint_callback,lr_monitor],'logger':logger}
    cli = LightningCLI(
                       model_class=CoolSystem,
                       trainer_defaults=trainer_defaults,
                       )
    
    
if __name__ == '__main__':
	#your code
    cli_main()