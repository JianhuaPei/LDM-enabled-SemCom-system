import argparse
import glob
import os
import numpy as np
import pandas as pd
from torchvision import datasets, transforms
import torch
#from model import ConvVAE
from model_new import ConvVAE
from Config import dic_obj as opt
from sklearn import preprocessing
from PIL import Image
from torch.utils.data import Dataset
from diffusers import AutoencoderKL  
from pytorch_msssim import ssim, ms_ssim
from SSIM_loss import SSIM, MS_SSIM
import torch.nn.functional as F
import lpips
import matplotlib.pyplot as plt
print(opt)


class Dataset_m(Dataset):

    def __init__(self, root, transform):
        self.root = root
        self.transform = transform
        self.data_list = glob.glob(f'{self.root}/*jpg')
        pass
    def __getitem__(self, index):
        data = self.data_list[index]
        data = Image.open(data)
        data = self.transform(data)

        return data

    def __len__(self):
        return len(self.data_list)
    
def loss_fn(recon_x, x, mu, logvar, critiron):   # defining loss function for va-AE (loss= reconstruction loss + KLD (to analyse if we have normal distributon))

    
    BCE = ms_ssim((recon_x+1)/2, (x+1)/2, data_range=1)
    
    #BCE = critiron(recon_x, x)

    # source: Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # KLD is equal to 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp()).cuda()

    return -BCE + 0.0001*KLD, BCE, KLD    

if __name__ == '__main__':
    
    input_shape = ( opt.inputdata_size_C, opt.inputdata_size_H, opt.inputdata_size_W)

    #load image dataset
    device = "cuda"

    #load dataset
    data_path = 'D:\\Python\\SemCom\\saved_data\\archive_cat_dog\\dog_3\\'

    transforms_train = transforms.Compose([transforms.Resize(opt.inputdata_size_H), transforms.CenterCrop(opt.inputdata_size_W), transforms.PILToTensor(), transforms.ConvertImageDtype(torch.float), transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])]) 

    # use CelebA dataset
    my_dataset = Dataset_m(f'{data_path}', transform = transforms_train)


    train_loader = torch.utils.data.DataLoader(
        dataset = my_dataset,
        batch_size=opt.batch_size_CVAE,
        shuffle=True,
    )    
    

    '''
    device = 'cpu'
    vae = AutoencoderKL.from_pretrained("D:\Python\SemCom_LDM\saved_model\diffusion")
    vae = vae.to(device)
    for param in vae.parameters():
        param.requires_grad = False 
    vae.eval()     
    ''' 
    iteration = 0
    model = ConvVAE().cuda()
    critiron = torch.nn.MSELoss()
    optimizer  = torch.optim.Adam(model.parameters(), lr = opt.lr_VAE)
    for param in model.parameters():
        param.requires_grad = True 

    lpips_model = lpips.LPIPS(net='alex') 
    for param in lpips_model.parameters():
        param.requires_grad = False

    bce_list = []
        
    for epoch in range(opt.epochs_VAE):
        epoch_bce = [] 
        for step, input_data in enumerate(train_loader):

    
            target = input_data.reshape(input_data.shape[0],opt.inputdata_size_C, opt.inputdata_size_H, opt.inputdata_size_W).cuda()
            target_output, mu, logvar  = model(target)
           
            iteration = epoch*len(train_loader)+step
            optimizer.zero_grad()    
            loss, bce, kld = loss_fn(target_output, target, mu, logvar, critiron )
            epoch_bce.append(bce.item())
            loss.backward()
            optimizer.step()
        avg_bce = sum(epoch_bce) / len(epoch_bce)
        bce_list.append(-10 * np.log10(1 - avg_bce))

        print (
                "[Epoch %d/%d] [iteration %d] [Loss: %f] [BCE: %f] [KLD: %f]"
                %(epoch, opt.epochs_VAE, iteration, loss.item(), bce.item(), kld.item())
                )    
    plt.figure()
    plt.plot(range(opt.epochs_VAE), bce_list, linestyle='-', color='b', linewidth=2.0)
    plt.title('MS-SSIM of AFHQ over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('MS-SSIM (dB)')
    plt.xlim(0, opt.epochs_VAE)
    plt.savefig('d:\\Python\\SemCom\\ms_ssim_plot.svg')  
    plt.close()     
    #save encoder model 
    torch.save(model.state_dict(), 'd:\\Python\\SemCom\\saved_model\\CVAE_model_of_Dog_4.pth')
