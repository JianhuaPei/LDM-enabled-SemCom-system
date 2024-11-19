import argparse
import glob
import os
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import datasets, transforms
import torch
import lpips
from model import Generator_DC, Discriminator_DC, CVAE_Encoder, Adaptor_g
from torch.utils.data import Dataset
from Config import dic_obj as opt
from sklearn import preprocessing
from matplotlib import pyplot as plt
from SSIM_loss import SSIM, MS_SSIM
from pytorch_msssim import ssim, ms_ssim
import torch.nn.functional as F
print(opt)


def cacl_gradient_penalty(net_D, real, fake):
    t = torch.rand(real.size(0), 1, 1, 1)
    t = t.expand(real.size())

    interpolates = t * real + (1 - t) * fake
    interpolates.requires_grad_(True)
    disc_interpolates = net_D(interpolates)
    grad = torch.autograd.grad(
        outputs=disc_interpolates, inputs=interpolates,
        grad_outputs=torch.ones_like(disc_interpolates),
        create_graph=True, retain_graph=True)[0]

    grad_norm = torch.norm(torch.flatten(grad, start_dim=1), dim=1)
    loss_gp = torch.mean((grad_norm - 1) ** 2)
    return loss_gp


class Dataset_m(Dataset):

    def __init__(self, root, transform):
        self.root = root
        self.transform = transform
        self.data_list = glob.glob(f'{self.root}/{opt.year}/*/*jpg')
        pass
    def __getitem__(self, index):
        data = self.data_list[index]
        data = Image.open(data).convert('L')
        data = self.transform(data)

        return data

    def __len__(self):
        return len(self.data_list)
    

def psnr(x, y, max_val=1.0):
    mse = F.mse_loss(x, y)
    psnr = 10 * torch.log10((max_val**2) / mse)
    return psnr.item()



if __name__ == '__main__':
    
    input_shape = ( opt.inputdata_size_C, opt.inputdata_size_H, opt.inputdata_size_W)
    device = "cpu"

    G = Generator_DC(input_dim = opt.latent_dim)
    G.load_state_dict(torch.load(f'd:\\Python\\SemCom_LDM\\saved_model\\generator_dc_model_of_MNIST.pth'))
    for param in G.parameters():
        param.requires_grad = False  
    
    E = CVAE_Encoder()
    E.load_state_dict(torch.load(f'd:\\Python\\SemCom_LDM\\saved_model\\v_encoder_dc_model_of_MNIST.pth'))
    for param in E.parameters():
        param.requires_grad = False  


    adaptor_D = Discriminator_DC()
    adaptor_D.load_state_dict(torch.load(f'd:\\Python\\SemCom_LDM\\saved_model\\discriminator_dc_model_of_MNIST.pth'))    
    for param in adaptor_D.parameters():
        param.requires_grad = False 

    adaptor_D.output[0].weight.requires_grad = True
    adaptor_D.output[0].bias.requires_grad = True

    #load dataset RBG->gray
    # use other dataset to verify domain adapatation
    #data_path = 'd:\\Python\\SemCom_LDM\\saved_data\\DIV2K\\DIV2K_train_HR\\DIV2K_train_HR\\0010.png'
    #data_path = 'd:\\Python\\SemCom_LDM\\saved_data\\archive\\data\\input_1_1_3.jpg'
    #data = Image.open(data_path).convert('L')
    #transform = transforms.Compose([transforms.Resize((opt.inputdata_size_H, opt.inputdata_size_W)), transforms.ToTensor(), transforms.Normalize#(mean=[0.5], std=[0.5])  ])
    #data_tensor = transform(data)

    transform = transforms.Compose([transforms.Resize(opt.inputdata_size_H), transforms.ToTensor(), transforms.Normalize(mean=[0.5], std=[0.5])  ])
    #transform = transforms.Compose([transforms.Resize(opt.inputdata_size_H), transforms.CenterCrop(opt.inputdata_size_W), transforms.PILToTensor(), transforms.ConvertImageDtype(torch.float), transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])])    

    # use MNIST dataset
    test_dataset = datasets.FashionMNIST('d:\\Python\\SemCom_LDM\\saved_data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = 1, shuffle=True)

    for test_data,labels in test_loader:
        test_data,labels = test_data.to(device), labels.to(device)
        break    
 

    adaptor_g = Adaptor_g(input_size = opt.latent_dim)
    #intial parameters = 1
    '''
    for param in adaptor_g.modules():
        if isinstance(param, torch.nn.Linear):
            param.weight.data.fill_(1)
            param.bias.data.zero_()
    '''
    adaptor_g.weight_init(mean=0.0, std=0.02)
    
    #Optimizers 
    optimizer_G = torch.optim.Adam(adaptor_g.parameters(), lr=opt.lr_GAN_G_a)  #improvement 2 of WGAN: utilze RMSProp or SGD to replace Adam or momentum
    optimizer_D = torch.optim.RMSprop(filter(lambda p: p.requires_grad, adaptor_D.parameters()), lr=opt.lr_GAN_D)
    E.eval()
    G.eval()

    shot_num = 1
    critiron = torch.nn.MSELoss() 
    ssim_loss = SSIM()


    ms_ssim_data = np.zeros(1000)
    mse_data = np.zeros(1000)



    #new for output
    #new for output
    data_tensor_real = torch.Tensor(16, opt.inputdata_size_C, opt.inputdata_size_W, opt.inputdata_size_H)
    data_tensor_ori = torch.Tensor(16, opt.inputdata_size_C, opt.inputdata_size_W, opt.inputdata_size_H)
    data_tensor_adaptor = torch.Tensor(16, opt.inputdata_size_C, opt.inputdata_size_W, opt.inputdata_size_H)


    ssim_ori = np.zeros(16)
    mse_ori = np.zeros(16)
    psnr_ori = np.zeros(16)

    ssim_opt = np.zeros(16)
    mse_opt = np.zeros(16)
    psnr_opt = np.zeros(16)   


    # Training
    for step, (input_data,labels) in enumerate(test_loader):
        if step == 16:
            break       
        
            
        data_tensor = input_data

        z,mu,var = E(data_tensor)
        
        gen_data = G(z.reshape(shot_num, opt.latent_dim,1,1))
        data_tensor_real[step,:,:,:] = data_tensor
        data_tensor_ori[step,:,:,:] = gen_data


        ssim_ori[step] = ssim((gen_data+1)/2, (data_tensor+1)/2, data_range =1)
        mse_ori[step] = critiron(data_tensor, gen_data) 
        psnr_ori[step] = psnr(data_tensor, gen_data)

        ssim_data = np.zeros(1000)
        mse_data = np.zeros(1000)
        psnr_data = np.zeros(1000)        

        for epoch in range(opt.epochs_adaptor):
            real_data = data_tensor.reshape(shot_num, opt.inputdata_size_C, opt.inputdata_size_H, opt.inputdata_size_W)

            optimizer_G.zero_grad()
            gen_data = G(adaptor_g(z).reshape(shot_num, opt.latent_dim,1,1))

            #Adversarial loss
            loss_G = -ssim((gen_data+1)/2, (real_data+1)/2, data_range=1)
            ssim_data[epoch] = - 10*torch.log10(1- ssim((gen_data+1)/2, (real_data+1)/2, data_range=1))
            mse_data[epoch] = critiron(gen_data, real_data)
            psnr_data[epoch] = psnr(real_data, gen_data)
            loss_G.backward()
            optimizer_G.step() #update the parameters of generator

            print (
                 "[Epoch %d/%d]  [adaptor_g loss: %f]"
                    %(epoch, opt.epochs_adaptor, loss_G.item())
                    )
            
        ssim_opt[step] = ssim_data[999]
        mse_opt[step] = mse_data[999] 
        psnr_opt[step] = psnr_data[999]   
    
 
    print( "[psnr: %f] [ms_ssim: %f]  [lpips: %f]"
          % (np.mean(psnr_ori), np.mean(ssim_ori), np.mean(mse_ori)) )
    print( "[psnr: %f] [ms_ssim: %f]  [lpips: %f]"
          % (np.mean(psnr_opt[psnr_opt>10]), np.mean(ssim_opt[ssim_opt>5]), np.mean(mse_opt[mse_opt<0.1])) )

    '''
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





    '''
    critiron = torch.nn.BCELoss()

    
    for epoch in range(opt.epochs_adaptor):


        #Configure input
        real_data = data_tensor.reshape(shot_num, opt.inputdata_size_C, opt.inputdata_size_H, opt.inputdata_size_W)
        
        #Train Discriminator

        optimizer_D.zero_grad()
        z = torch.tensor(np.random.normal(0,1,(real_data.shape[0], opt.latent_dim))).float()
        #z = z.reshape(real_data.shape[0], opt.latent_dim,1,1)           
        true_of_D = torch.mean(adaptor_D(real_data))
        fake_data = G(adaptor_g(z).reshape(real_data.shape[0], opt.latent_dim,1,1)  )
        fake_of_D = torch.mean(adaptor_D(fake_data))
        loss_gp = cacl_gradient_penalty(adaptor_D, real_data, fake_data)
        loss_D = fake_of_D - true_of_D + opt.lambda_1*loss_gp
        loss_D.backward()
        optimizer_D.step() #update the parameters of D


        #Train Generator
        if epoch % opt.n_critic == 0:
            optimizer_G.zero_grad()
            z = torch.tensor(np.random.normal(0,1,(real_data.shape[0], opt.latent_dim))).float()
            #z = z.reshape(real_data.shape[0], opt.latent_dim,1,1) 
            fake_data = G(adaptor_g(z).reshape(real_data.shape[0], opt.latent_dim,1,1)  )
            loss_G = -torch.mean(adaptor_D(fake_data))
            loss_G.backward()
            optimizer_G.step() #update the parameters of G

        print (
                 "[Epoch %d/%d]  [adaptor_D loss: %f] [adaptor_g loss: %f]"
                    %(epoch, opt.epochs_adaptor, loss_D.item(),loss_G.item())
                    )
           
    torch.save(adaptor_g.state_dict(), 'd:\\Python\\SemCom_LDM\\saved_model\\adaptor_model_of_MNIST.pth')  
    '''    


    '''
    # test 
    a_g = Adaptor_g(input_size = opt.latent_dim)
    a_g.load_state_dict(torch.load(f'd:\\Python\\SemCom_LDM\\saved_model\\adaptor_model_of_MNIST.pth'))    
    for param in a_g.parameters():
        param.requires_grad = False 

    a_g.eval()    

    fig = plt.figure(figsize=(1,1), constrained_layout = True)
    gs = fig.add_gridspec(1,3)

    f_ax = fig.add_subplot(gs[0,0])
    f_ax.imshow(test_data.squeeze().cpu().numpy(), cmap="gray")
    f_ax.axis("off")
    data_recons = G(E(test_data).reshape(real_data.shape[0], opt.latent_dim, 1, 1))
    f_ax = fig.add_subplot(gs[0,1])
    f_ax.imshow(data_recons.squeeze().cpu().numpy(), cmap="gray") 
    f_ax.axis("off")
    data_recons_ada = G(a_g(E(test_data)[:,-1,:]).reshape(real_data.shape[0], opt.latent_dim, 1, 1))   
    f_ax = fig.add_subplot(gs[0,2])
    f_ax.imshow(data_recons_ada.squeeze().cpu().numpy(), cmap="gray") 
    f_ax.axis("off")
    plt.savefig('d:\\Python\\SemCom_LDM\\saved_data\\test_mnist_ada.svg')
    '''










    