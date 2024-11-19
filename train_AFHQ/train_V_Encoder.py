import argparse
import glob
import os
import numpy as np
import pandas as pd
from torchvision import datasets, transforms
import torch
from model import Generator_DC
from Config import dic_obj as opt
from sklearn import preprocessing
from PIL import Image
from torch.utils.data import Dataset
from matplotlib import pyplot as plt
from model_new import CVAE_Encoder
from pytorch_msssim import ssim, ms_ssim




def loss_fn(recon_x, x, mu, logvar, critiron):   # defining loss function for va-AE (loss= reconstruction loss + KLD (to analyse if we have normal distributon))

    
    BCE = ms_ssim((recon_x+1)/2, (x+1)/2, data_range=1)

    # source: Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # KLD is equal to 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    return -BCE + 0.01*KLD, BCE, KLD   

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



if __name__ == '__main__':
    input_shape = ( opt.inputdata_size_C, opt.inputdata_size_H, opt.inputdata_size_W)




    #load dataset
    data_path = 'D:\\Python\\SemCom\\saved_data\\archive_cat_dog\\dog_4'

    transforms_train = transforms.Compose([transforms.Resize(opt.inputdata_size_H), transforms.CenterCrop(opt.inputdata_size_W), transforms.PILToTensor(), transforms.ConvertImageDtype(torch.float), transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])]) 
    my_dataset = Dataset_m(f'{data_path}', transform = transforms_train)
    train_loader = torch.utils.data.DataLoader(
        dataset = my_dataset,
        batch_size=opt.batch_size_VAE,
        shuffle=True,
    )  

    G=Generator_DC(input_dim = opt.latent_dim)
    G.load_state_dict(torch.load(f'd:\\Python\\SemCom\\saved_model\\generator_dc_model_of_Dog.pth'))

    for param in G.parameters():
        param.requires_grad = False
    G.eval()    

    z =  torch.tensor(np.random.normal(0,1,(4, opt.latent_dim))).float()    
    z =z.reshape(4, opt.latent_dim,1,1)
    x= G(z)
    fig = plt.figure(figsize=(10,10), constrained_layout = True)
    gs = fig.add_gridspec(1,4)    
    for i in range(4):
        f_ax = fig.add_subplot(gs[0,i])
        f_ax.imshow((((x[i].squeeze().permute(1,2,0)+1)*127.5).cpu().numpy()).astype(np.uint8))
        f_ax.axis("off")  
    plt.savefig('d:\\Python\\SemCom\\saved_data\\test_class_256.svg')  

    iteration = 0
    model = CVAE_Encoder()
    critiron = torch.nn.MSELoss()
    optimizer  = torch.optim.Adam(model.parameters(), lr = opt.lr_VAE)

    for epoch in range(opt.epochs_VAE):
        for step, input_data in enumerate(train_loader):

    
            target = input_data.reshape(input_data.shape[0], opt.inputdata_size_C, opt.inputdata_size_H, opt.inputdata_size_W)
            target_emb = model(input_data).reshape(input_data.shape[0], opt.latent_dim)
            
            iteration = epoch*len(train_loader)+step
            with torch.no_grad():
                gen_data = G(z=target_emb, y= y)
            with torch.set_grad_enabled(True):
                optimizer.zero_grad()    
                gen_data.requires_grad=True
                loss = critiron(gen_data, target)
                loss.backward()
                optimizer.step()

        print (
                "[Epoch %d/%d] [iteration %d] [Loss: %f] "
                %(epoch, opt.epochs_VAE, iteration, loss.item())
                )    
         
    #save encoder model 
    torch.save(model.state_dict(), 'd:\\Python\\SemCom_LDM\\saved_model\\v_encoder_dc_model_of_pretrain_512.pth')    


