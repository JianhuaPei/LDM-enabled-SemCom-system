import argparse
import glob
import os
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import datasets, transforms
import torch
import lpips
from model import Generator, Discriminator, V_Encoder, Adaptor_g
from model_new import ConvVAE, CVAE_Encoder
from torch.utils.data import Dataset
from Config import dic_obj as opt
from sklearn import preprocessing
from matplotlib import pyplot as plt
from SSIM_loss import SSIM, MS_SSIM
from pytorch_msssim import ssim, ms_ssim
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



if __name__ == '__main__':
    
    input_shape = ( opt.inputdata_size_C, opt.inputdata_size_H, opt.inputdata_size_W)
    device = "cpu"

    '''
    G = Generator(input_dim = opt.latent_dim)
    G.load_state_dict(torch.load(f'd:\\Python\\SemCom_LDM\\saved_model\\generator_model_of_CIFAR10.pth'))
    for param in G.parameters():
        param.requires_grad = False  

    E = V_Encoder()
    E.load_state_dict(torch.load(f'd:\\Python\\SemCom_LDM\\saved_model\\v_encoder_model_of_CIFAR10.pth'))
    for param in E.parameters():
        param.requires_grad = False  


    adaptor_D = Discriminator()
    adaptor_D.load_state_dict(torch.load(f'd:\\Python\\SemCom_LDM\\saved_model\\discriminator_model_of_CIFAR10.pth'))    
    for param in adaptor_D.parameters():
        param.requires_grad = False 

    adaptor_D.fc.weight.requires_grad = True
    adaptor_D.fc.bias.requires_grad = True
    '''

    #load dataset RBG->gray
    # use other dataset to verify domain adapatation
    #load dataset
    data_path = 'd:\\Python\\SemCom\\saved_data\\archive_cat_dog\\cat_1'

    transforms_train = transforms.Compose([transforms.Resize(opt.inputdata_size_H), transforms.CenterCrop(opt.inputdata_size_W), transforms.PILToTensor(), transforms.ConvertImageDtype(torch.float), transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])]) 
    my_dataset = Dataset_m(f'{data_path}', transform = transforms_train)
    train_loader = torch.utils.data.DataLoader(
        dataset = my_dataset,
        batch_size=1,
        shuffle=True,
    ) 
    VAE = ConvVAE().cuda()
    VAE.load_state_dict(torch.load(f'd:\\Python\\SemCom\\saved_model\\CVAE_model_of_Dog.pth'))
    for param in VAE.parameters():
        param.requires_grad = False  
    VAE.eval()

    E_new=CVAE_Encoder().cuda()
    E_new.load_state_dict(torch.load(f'd:\\Python\\SemCom\\saved_model\\robust_encoder_model_of_Dog.pth'))
    for param in E_new.parameters():
        param.requires_grad = False  
    E_new.eval()              



    adaptor_g = Adaptor_g(input_size = opt.latent_dim).cuda()
    

    optimizer_G = torch.optim.Adam(adaptor_g.parameters(), lr=opt.lr_GAN_G)
    #intial parameters = 1
    '''
    for param in adaptor_g.modules():
        if isinstance(param, torch.nn.Linear):
            param.weight.data.fill_(1)
            param.bias.data.zero_()
    '''
    '''  
    #Optimizers 
    optimizer_G = torch.optim.Adam(adaptor_g.parameters(), lr=opt.lr_GAN_G)  #improvement 2 of WGAN: utilze RMSProp or SGD to replace Adam or momentum
    optimizer_D = torch.optim.RMSprop(filter(lambda p: p.requires_grad, adaptor_D.parameters()), lr=opt.lr_GAN_D)
    E.eval()
    G.eval()
    '''

    shot_num = 1
    critiron = torch.nn.MSELoss() 
    ssim_loss = SSIM()

    #new for output
    data_tensor_real = torch.Tensor(5, opt.inputdata_size_C, opt.inputdata_size_W, opt.inputdata_size_H)
    data_tensor_ori = torch.Tensor(5, opt.inputdata_size_C, opt.inputdata_size_W, opt.inputdata_size_H)
    data_tensor_adaptor = torch.Tensor(5, opt.inputdata_size_C, opt.inputdata_size_W, opt.inputdata_size_H)


    # Training
    for step, input_data in enumerate(train_loader):
        if step == 5:
            break       
        
             
        data_tensor = input_data.cuda()
        z,mu,var = E_new(data_tensor)
        gen_data = VAE.decoder(z)
        data_tensor_real[step,:,:,:] = data_tensor
        data_tensor_ori[step,:,:,:] = gen_data

        for epoch in range(opt.epochs_adaptor):
            real_data = data_tensor.reshape(shot_num, opt.inputdata_size_C, opt.inputdata_size_H, opt.inputdata_size_W)

            optimizer_G.zero_grad()
            gen_data = VAE.decoder(adaptor_g(z))

            #Adversarial loss
            loss_G = -ms_ssim((gen_data+1)/2, (real_data+1)/2, data_range=1)
            loss_G.backward()
            optimizer_G.step() #update the parameters of generator

            print (
                 "[Epoch %d/%d]  [adaptor_g loss: %f]"
                    %(epoch, opt.epochs_adaptor, loss_G.item())
                    )
            
        gen_data = VAE.decoder(adaptor_g(z))
        data_tensor_adaptor[step,:,:,:] = gen_data
        
    fig = plt.figure(figsize=(10,10), constrained_layout = True)
    gs = fig.add_gridspec(5,3)
    for i in range(5):
        f_ax = fig.add_subplot(gs[i,0])
        f_ax.imshow(data_tensor_real[i][0].detach().numpy(), cmap="gray")
        f_ax.axis("off")
        f_ax = fig.add_subplot(gs[i,1])
        f_ax.imshow(data_tensor_ori[i][0].detach().numpy(), cmap="gray")
        f_ax.axis("off")
        f_ax = fig.add_subplot(gs[i,2])
        f_ax.imshow(data_tensor_adaptor[i][0].detach().numpy(), cmap="gray")
        f_ax.axis("off")
   
    plt.savefig('d:\\Python\\SemCom_LDM\\saved_data\\test_adaptor_mnist.svg') 


        


        
        
 






    '''
    for epoch in range(opt.epochs_adaptor):


        #Configure input
        real_data = data_tensor.reshape(shot_num, opt.inputdata_size_C, opt.inputdata_size_H, opt.inputdata_size_W)
        
        #Train Discriminator
        optimizer_D.zero_grad()

        #Sample noise as generator input
        z = torch.tensor(np.random.normal(0,1,(real_data.shape[0], opt.latent_dim))).float()
        #z = E(real_data)[:,-1,:]

        #Generate a batch of inputdata
        fake_data = G(adaptor_g(z)[:,-1,:]).detach()
            

        fake_of_D = torch.mean(adaptor_D(fake_data))
        true_of_D = torch.mean(adaptor_D(real_data))
        loss_D = fake_of_D - true_of_D  # improvement 3 of WGAN: the loss of G and D do not take log

        loss_D.backward()
        optimizer_D.step() #update the parameters of D

        # Clip weights of discriminator
        
        for p in adaptor_D.parameters():
            p.data.clamp_(-opt.clip_value, opt.clip_value)  #improvement 4 of WGAN: clip
        
            
        # Train the generator every n_critic iterations
        if epoch % 2 == 0:

            # Train Generator
            z = torch.tensor(np.random.normal(0,1,(real_data.shape[0], opt.latent_dim))).float()
            #z = E(real_data)[:,-1,:]
            optimizer_G.zero_grad()

            gen_data = G(adaptor_g(z)[:,-1,:])

            #Adversarial loss
            loss_G = -torch.mean(adaptor_D(gen_data))
            loss_G.backward()
            optimizer_G.step() #update the parameters of generator

        print (
                 "[Epoch %d/%d]  [adaptor_D loss: %f] [adaptor_g loss: %f]"
                    %(epoch, opt.epochs_adaptor, loss_D.item(),loss_G.item())
                    )
           
    torch.save(adaptor_g.state_dict(), 'd:\\Python\\SemCom_LDM\\saved_model\\adaptor_model_of_MNIST.pth')  
    '''


    '''
    # test 
    a_g = Adaptor_g(input_size = opt.latent_dim)
    a_g.load_state_dict(torch.load(f'd:\\Python\\SemCom_LDM\\saved_model\\adaptor_model_of_CIFAR10.pth'))    
    for param in a_g.parameters():
        param.requires_grad = False 

    a_g.eval()    

    fig = plt.figure(figsize=(1,1), constrained_layout = True)
    gs = fig.add_gridspec(1,3)

    f_ax = fig.add_subplot(gs[0,0])
    f_ax.imshow((((data_tensor.squeeze().permute(1,2,0)+1)*127.5).cpu().numpy()).astype(np.uint8))
    f_ax.axis("off")
    data_recons = G(E(data_tensor)[:,-1,:])
    f_ax = fig.add_subplot(gs[0,1])
    f_ax.imshow((((data_recons.squeeze().permute(1,2,0)+1)*127.5).cpu().numpy()).astype(np.uint8)) 
    f_ax.axis("off")
    data_recons_ada = G(a_g(E(data_tensor)[:,-1,:])[:,-1,:])   
    f_ax = fig.add_subplot(gs[0,2])
    f_ax.imshow((((data_recons_ada.squeeze().permute(1,2,0)+1)*127.5).cpu().numpy()).astype(np.uint8)) 
    f_ax.axis("off")
    plt.savefig('d:\\Python\\SemCom_LDM\\saved_data\\test_cifar10_ada.svg')
    '''










    