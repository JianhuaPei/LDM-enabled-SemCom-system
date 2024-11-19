import os
import torch

os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12345'
torch.distributed.init_process_group(backend='gloo', rank=0, world_size=1)

from torchvision import datasets, transforms

from model import Generator_DC, Discriminator_DC, CVAE_Encoder, Adaptor_g
from torch.utils.data import Dataset
from Config import dic_obj as opt
from sklearn import preprocessing
from matplotlib import pyplot as plt


from networks import VEPrecond
from loss import VELoss

import yaml
import argparse
from diffusion import Diffusion
import pytorch_lightning as pl
from ema import EMA, EMAModelCheckpoint
from torch.utils.data import DataLoader
from data import get_dataset
from pytorch_lightning.strategies.ddp import DDPStrategy
import click



def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


@click.command()
@click.option('--cfg', default='D:\\Python\\SemCom\\train_MNIST\\config.yml', help='Configuration File')
def main(cfg):
    with open(cfg, "r") as f:
        cfg = yaml.safe_load(f)
        cfg = dict2namespace(cfg)
    filepath = 'd:\\Python\\SemCom\\saved_model'

    ckpt_callback = EMAModelCheckpoint(dirpath=filepath, save_top_k=1, monitor="train_loss", save_last=True, filename='{epoch}-{train_loss:.8f}', every_n_train_steps=None,)
    ema_callback = EMA(decay=cfg.model.ema_rate)
    callbacks = [ckpt_callback, ema_callback]

    model = Diffusion(cfg)

    transform = transforms.Compose([transforms.Resize(opt.inputdata_size_H), transforms.ToTensor(), transforms.Normalize(mean=[0.5], std=[0.5])  ])
    #transform = transforms.Compose([transforms.Resize(opt.inputdata_size_H), transforms.CenterCrop(opt.inputdata_size_W), transforms.PILToTensor(), transforms.ConvertImageDtype(torch.float), transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])])

    # use MNIST dataset
    dataset = datasets.MNIST('d:\\Python\\SemCom\\saved_data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size = opt.batch_size_diffusion, shuffle=True)
    dataset = datasets.MNIST('d:\\Python\\SemCom\\saved_data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(dataset, batch_size = opt.batch_size_diffusion, shuffle=True)    


    trainer = pl.Trainer(
        callbacks=callbacks,
        precision=cfg.training.precision,
        #max_steps=cfg.training.max_steps,
        max_epochs=20,
        accumulate_grad_batches=cfg.training.accumulate_grad_batches,
        accelerator="gpu", 
        devices=[0],
        #limit_val_batches=1,
        limit_train_batches=0.5,
        gradient_clip_val=cfg.optim.grad_clip,
        benchmark=True,
        strategy = DDPStrategy(find_unused_parameters=False),
    )

    # Train
    trainer.fit(model, train_dataloaders=train_loader)

def normalize_fun(z_tensor):
    x_normalized = torch.zeros_like(z_tensor)

    # 按batch归一化
    for i in range(z_tensor.size(0)):
        batch = z_tensor[i]  # 取出第i个batch
        min_val = batch.min()
        max_val = batch.max()

        batch_normalized = (batch - min_val) / (max_val - min_val)
        batch_normalized = batch_normalized * 2 - 1

        x_normalized[i] = batch_normalized    
    return x_normalized



if __name__ == '__main__':

    #main()

    input_shape = ( opt.inputdata_size_C, opt.inputdata_size_H, opt.inputdata_size_W)
    device = "cuda"

    G = Generator_DC(input_dim = opt.latent_dim).cuda()
    G.load_state_dict(torch.load(f'd:\\Python\\SemCom\\saved_model\\generator_dc_model_of_MNIST.pth'))
    for param in G.parameters():
        param.requires_grad = False  
    G.eval()
    
    E = CVAE_Encoder().cuda()
    E.load_state_dict(torch.load(f'd:\\Python\\SemCom\\saved_model\\v_encoder_dc_model_of_MNIST.pth'))
    for param in E.parameters():
        param.requires_grad = False  
    E.eval()

    #train VE based diffusion model random shedule, noise variance 

    VE_model = VEPrecond(img_resolution = 8, img_channels = opt.inputdata_size_C).cuda()

    loss_fn = VELoss()

    transform = transforms.Compose([transforms.Resize(opt.inputdata_size_H), transforms.ToTensor(), transforms.Normalize(mean=[0.5], std=[0.5])  ])
    #transform = transforms.Compose([transforms.Resize(opt.inputdata_size_H), transforms.CenterCrop(opt.inputdata_size_W), transforms.PILToTensor(), transforms.ConvertImageDtype(torch.float), transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])])

    # use MNIST dataset
    dataset = datasets.MNIST('d:\\Python\\SemCom\\saved_data', train=False, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size = opt.batch_size_diffusion, shuffle=True)
    optimizer = torch.optim.RAdam(VE_model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay, betas=opt.betas, eps=opt.eps)   


    iteration = 0
    
        
    for epoch in range(opt.epochs_VE_diffusion):
        for step, (input_data, labels) in enumerate(train_loader):
            input_data =input_data.cuda()
            
            iteration = epoch*len(train_loader)+step
            target_z, mu, var = E(input_data)
            target_z = target_z.reshape(input_data.shape[0], 1, 8, 8) 
            target_z = normalize_fun(target_z)




            optimizer.zero_grad()
            loss =    loss_fn(net = VE_model, images = target_z, labels=None).mean()

            loss.backward()
            optimizer.step()


        print (
                "[Epoch %d/%d] [iteration %d] [Loss: %f] "
                %(epoch, opt.epochs_VE_diffusion, iteration, loss.item())
                )   


    #save diffusion model 
    torch.save(VE_model.state_dict(), 'd:\\Python\\SemCom\\saved_model\\VE_diffusion_AWGN_model_of_MNIST.pth')


    #train CD model

    


