import argparse
import glob
import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import pandas as pd
from torchvision import datasets, transforms
import torch
from model import Generator_DC, Discriminator_DC
from Config import dic_obj as opt
from sklearn import preprocessing
print(opt)


def cacl_gradient_penalty(net_D, real, fake):
    t = torch.rand(real.size(0), 1, 1, 1).cuda()
    t = t.expand(real.size())

    interpolates = t * real + (1 - t) * fake
    interpolates.requires_grad_(True)
    disc_interpolates = net_D(interpolates)
    grad = torch.autograd.grad(
        outputs=disc_interpolates, inputs=interpolates,
        grad_outputs=torch.ones_like(disc_interpolates).cuda(),
        create_graph=True, retain_graph=True)[0]

    grad_norm = torch.norm(torch.flatten(grad, start_dim=1), dim=1)
    loss_gp = torch.mean((grad_norm - 1) ** 2)
    return loss_gp



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


if __name__=='__main__':

    input_shape = ( opt.inputdata_size_C, opt.inputdata_size_H, opt.inputdata_size_W)
    cuda = True if torch.cuda.is_available() else False
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


    generator = Generator_DC(input_dim=opt.latent_dim).cuda()
    discriminator = Discriminator_DC().cuda()

    generator.weight_init(mean=0.0, std=0.02)
    discriminator.weight_init(mean=0.0, std=0.02)

    #Optimizers 
    optimizer_G = torch.optim.Adam(generator.parameters(),lr=opt.lr_GAN_G, betas=(0.5, 0.999))  #improvement 2 of WGAN: utilze RMSProp or SGD to replace Adam or momentum
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr_GAN_D, betas=(0.5,0.999))

    # Training

    for epoch in range(opt.epochs_GAN):

        for step, input_data in enumerate(train_loader):

            #Configure input
            real_data = input_data.cuda()
            iteration = epoch*len(train_loader)+step

            #Train Discriminator
            optimizer_D.zero_grad()
            z = torch.tensor(np.random.normal(0,1,(real_data.shape[0], opt.latent_dim))).float().cuda()
            z = z.reshape(real_data.shape[0], opt.latent_dim,1,1)           
            true_of_D = torch.mean(discriminator(real_data))
            fake_data = generator(z)
            fake_of_D = torch.mean(discriminator(fake_data))
            loss_gp = cacl_gradient_penalty(discriminator, real_data, fake_data)
            loss_D = fake_of_D - true_of_D + opt.lambda_1*loss_gp
            loss_D.backward()
            optimizer_D.step() #update the parameters of D
            #for p in discriminator.parameters():
            #    p.data.clamp_(-opt.clip_value,opt.clip_value) 

            #Train Generator
            if step % opt.n_critic == 0:
                optimizer_G.zero_grad()
                z = torch.tensor(np.random.normal(0,1,(real_data.shape[0], opt.latent_dim))).float().cuda()
                z = z.reshape(real_data.shape[0], opt.latent_dim,1,1) 
                fake_data = generator(z)
                loss_G = -torch.mean(discriminator(fake_data))
                loss_G.backward()
                optimizer_G.step() #update the parameters of G

            
        print (
                        "[Epoch %d/%d] [iteration %d] [D loss: %f] [G loss: %f]"
                        %(epoch, opt.epochs_GAN, iteration, loss_D.item(),loss_G.item())
                        )

                
    torch.save(generator.state_dict(), 'd:\\Python\\SemCom\\saved_model\\generator_dc_model_of_DIV2K.pth')    
    torch.save(discriminator.state_dict(), 'd:\\Python\\SemCom\\saved_model\\discriminator_dc_model_of_DIV2K.pth') 