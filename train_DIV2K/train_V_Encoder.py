import argparse
import glob
import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import pandas as pd
from torchvision import datasets, transforms
import torch
from model import Generator_DC, Discriminator_DC, Encoder
from Config import dic_obj as opt
from sklearn import preprocessing
from matplotlib import pyplot as plt
print(opt)

class Dataset_m(Dataset):

    def __init__(self, root, transform):
        self.root = root
        self.transform = transform
        self.data_list = glob.glob(f'{self.root}/*png')
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

    #load image dataset

    #load dataset

    data_path = 'd:\\Python\\SemCom\saved_data\\DIV2K_valid_HR\\DIV2K_valid_HR'
    #transform = transforms.Compose([transforms.Resize(opt.inputdata_size_H), transforms.ToTensor(), transforms.Normalize(mean=[0.5], std=[0.5])  ])
    transforms_train = transforms.Compose([transforms.Resize(opt.inputdata_size_H), transforms.CenterCrop(opt.inputdata_size_W), transforms.PILToTensor(), transforms.ConvertImageDtype(torch.float), transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])]) 

    # use DIV2K dataset
    my_dataset = Dataset_m(f'{data_path}', transform = transforms_train)


    train_loader = torch.utils.data.DataLoader(
        dataset = my_dataset,
        batch_size=opt.batch_size_GAN,
        shuffle=True,
    )    

    
    model = Encoder().cuda()
    critiron = torch.nn.MSELoss()
    optimizer  = torch.optim.Adam(model.parameters(), lr = opt.lr_VAE)
    G=Generator_DC(input_dim = opt.latent_dim).cuda()
    G.load_state_dict(torch.load(f'D:\Python\SemCom\saved_model\generator_dc_model_of_DIV2K.pth'))
    for param in G.parameters():
        param.requires_grad = False 


    z =  torch.tensor(np.random.normal(0,1,(4, opt.latent_dim))).float().cuda()    
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
        
    for epoch in range(opt.epochs_VAE):
        for step, input_data in enumerate(train_loader):

            optimizer.zero_grad()
            input_data = input_data.cuda()

            target = input_data.reshape(input_data.shape[0],opt.inputdata_size_C, opt.inputdata_size_H, opt.inputdata_size_W).cuda()
            target_emb = model(input_data)
            target_emb = target_emb.reshape(input_data.shape[0], opt.latent_dim, 1, 1)
            iteration = epoch*len(train_loader)+step
            
            G.eval()
            gen_data = G(target_emb)
            loss = critiron(gen_data, target)
            loss.backward()
            optimizer.step()
         
            
            
        print (
                        "[Epoch %d/%d] [iteration %d] [Loss: %f] "
                        %(epoch, opt.epochs_VAE, iteration, loss.item())
                )


    #save encoder model 
    torch.save(model.state_dict(), 'd:\\Python\\SemCom_LDM\\saved_model\\v_encoder_dc_model_of_DIV2K.pth')
