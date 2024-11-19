import argparse
import glob
import os
import numpy as np
import pandas as pd
from torchvision import datasets, transforms
import torch
from model import Generator_DC, Discriminator_DC, CVAE_Encoder
from Config import dic_obj as opt
from sklearn import preprocessing
from pytorch_msssim import ssim, ms_ssim
print(opt)



def loss_fn(recon_x, x, mu, logvar, critiron):   # defining loss function for va-AE (loss= reconstruction loss + KLD (to analyse if we have normal distributon))

    
    BCE = ssim((recon_x+1)/2, (x+1)/2, data_range=1)

    # source: Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # KLD is equal to 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    return -BCE , BCE, KLD   

if __name__ == '__main__':
    
    input_shape = ( opt.inputdata_size_C, opt.inputdata_size_H, opt.inputdata_size_W)

    #load image dataset

    #load dataset

    transform = transforms.Compose([transforms.Resize(opt.inputdata_size_H), transforms.ToTensor(), transforms.Normalize(mean=[0.5], std=[0.5])  ])
    #transform = transforms.Compose([transforms.Resize(opt.inputdata_size_H), transforms.CenterCrop(opt.inputdata_size_W), transforms.PILToTensor(), transforms.ConvertImageDtype(torch.float), transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])])

    # use MNIST dataset
    dataset = datasets.MNIST('d:\\Python\\SemCom\\saved_data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size = opt.batch_size_VAE, shuffle=True)
    
    model = CVAE_Encoder().cuda()
    critiron = torch.nn.MSELoss()
    optimizer  = torch.optim.Adam(model.parameters(), lr = opt.lr_VAE)
    G=Generator_DC(input_dim = opt.latent_dim).cuda()
    G.load_state_dict(torch.load(f'd:\\Python\\SemCom\\saved_model\\generator_dc_model_of_MNIST.pth'))
    for param in G.parameters():
        param.requires_grad = False    
    
    iteration = 0
    
        
    for epoch in range(opt.epochs_VAE):
        for step, (input_data, labels) in enumerate(train_loader):

            optimizer.zero_grad()


            target = input_data.reshape(input_data.shape[0],opt.inputdata_size_C, opt.inputdata_size_H, opt.inputdata_size_W).cuda()
            target_emb, mu, logvar  = model(target)
            target_emb = target_emb.reshape(input_data.shape[0], opt.latent_dim, 1, 1)
            iteration = epoch*len(train_loader)+step
            
            gen_data = G(target_emb)
            loss, bce, kld = loss_fn(gen_data, target, mu, logvar, critiron )
            loss.backward()
            optimizer.step()
            
            
        print (
                "[Epoch %d/%d] [iteration %d] [Loss: %f] [BCE: %f] [KLD: %f]"
                %(epoch, opt.epochs_VAE, iteration, loss.item(), bce.item(), kld.item())
                )   

            
         
    #save encoder model 
    torch.save(model.state_dict(), 'd:\\Python\\SemCom\\saved_model\\v_encoder_dc_model_of_MNIST.pth')
