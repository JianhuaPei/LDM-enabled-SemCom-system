import argparse
import glob
import os
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import datasets, transforms
import torch
import lpips
from model import Generator, Discriminator, V_Encoder
from model_new import Adaptor_g
from model_new import ConvVAE, CVAE_Encoder
from torch.utils.data import Dataset
from Config import dic_obj as opt
from sklearn import preprocessing
from matplotlib import pyplot as plt
from SSIM_loss import SSIM, MS_SSIM
from pytorch_msssim import ssim, ms_ssim
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from diffusers import AutoencoderKL 
import torch.nn.functional as F

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
    

def lpips_cal(x,y,lpips_loss):
    x = x
    y =y
    lpips_output = lpips_loss(x,y)
    #lpips_output = F.mse_loss(x,y)
    return lpips_output



def ms_ssim_cal(x,y):
    ms_ssim_output = ms_ssim((x+1)/2, (y+1)/2, data_range=1)
    ms_ssim_output = -10*torch.log10(1-ms_ssim_output)
    return ms_ssim_output    

def psnr(x, y, max_val=1.0):
    mse = F.mse_loss(x, y)
    psnr = 10 * torch.log10((max_val**2) / mse)
    return psnr.item()
    


def normalize_fun(x):
    # 初始化归一化后的Tensor
    x_normalized = torch.zeros_like(x)
    
    # 初始化最大值和最小值列表
    max_vals = []
    min_vals = []
    
    # 按通道进行归一化
    for c in range(x.size(1)):  # 遍历所有通道
        # 选择当前通道的数据
        x_channel = x[:, c, :, :]

        # 计算当前通道的最小值和最大值
        min_val = torch.min(x_channel)
        max_val = torch.max(x_channel)
        min_vals.append(min_val.item())
        max_vals.append(max_val.item())

        # 归一化到[0,1]
        x_channel_normalized = (x_channel - min_val) / (max_val - min_val)

        # 再归一化到[-1,1]
        x_channel_normalized = (x_channel_normalized * 2) - 1

        # 将归一化后的通道数据放回
        x_normalized[:, c, :, :] = x_channel_normalized

    # 返回归一化后的Tensor、最大值和最小值
    return x_normalized.cuda(), max_vals, min_vals

def denormalize_tensor(x_normalized, max_vals, min_vals):
    x_denormalized = torch.zeros_like(x_normalized).cuda()
    
    # 反变换
    for c in range(x_normalized.size(1)):  # 遍历所有通道
        x_channel_normalized = x_normalized[:, c, :, :]
        
        # 反归一化到[0,1]
        x_channel = (x_channel_normalized + 1) / 2 * (max_vals[c] - min_vals[c]) + min_vals[c]
        
        # 将反归一化后的通道数据放回
        x_denormalized[:, c, :, :] = x_channel
        
    return x_denormalized




if __name__ == '__main__':
    
    input_shape = ( opt.inputdata_size_C, opt.inputdata_size_H, opt.inputdata_size_W)
    device = "cuda"

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
    data_path = 'd:\\Python\\SemCom\\saved_data\\DIV2K_test_HR\\DIV2K_valid_HR'

    transforms_train = transforms.Compose([transforms.Resize(opt.inputdata_size_H), transforms.CenterCrop(opt.inputdata_size_W), transforms.PILToTensor(), transforms.ConvertImageDtype(torch.float), transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])]) 
    my_dataset = Dataset_m(f'{data_path}', transform = transforms_train)
    train_loader = torch.utils.data.DataLoader(
        dataset = my_dataset,
        batch_size=1,
        shuffle=True,
    ) 

    '''
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
    '''

    vae = AutoencoderKL.from_pretrained("D:\\Python\\SemCom\\saved_model\\diffusion")
    vae = vae.to(device)
    for param in vae.parameters():
        param.requires_grad = False 
    vae.eval()           


    lpips_loss = LearnedPerceptualImagePatchSimilarity(net_type='vgg')
    lpips_loss = lpips_loss.to(device)
    for param in lpips_loss.parameters():
        param.requires_grad = False 
    lpips_loss.eval() 

    adaptor_g = Adaptor_g().cuda()

    ms_ssim_data = np.zeros(200)
    lpips_data = np.zeros(200)
    psnr_data = np.zeros(200)


    ms_ssim_ori = np.zeros(16)
    lpips_ori = np.zeros(16)
    psnr_ori = np.zeros(16)

    ms_ssim_opt = np.zeros(16)
    lpips_opt = np.zeros(16)
    psnr_opt = np.zeros(16)    

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

    #new for output
    data_tensor_real = torch.Tensor(16, opt.inputdata_size_C, opt.inputdata_size_W, opt.inputdata_size_H)
    data_tensor_ori = torch.Tensor(16, opt.inputdata_size_C, opt.inputdata_size_W, opt.inputdata_size_H)
    data_tensor_adaptor = torch.Tensor(16, opt.inputdata_size_C, opt.inputdata_size_W, opt.inputdata_size_H)


    # Training
    for step, input_data in enumerate(train_loader):
        if step == 16:
            break       
        
        data_tensor = input_data.cuda()
        z= vae.encode(data_tensor).latent_dist.sample() 
        gen_data = vae.decode(z).sample
        ms_ssim_ori[step] = ms_ssim_cal(gen_data, data_tensor)
        lpips_ori[step] = lpips_cal(data_tensor, gen_data.clamp(-1.0,1.0), lpips_loss) 
        psnr_ori[step] = psnr(data_tensor, gen_data)


        data_tensor_real[step,:,:,:] = data_tensor
        data_tensor_ori[step,:,:,:] = gen_data

        z, max_val, min_val = normalize_fun(z)

        for epoch in range(opt.epochs_adaptor):
            real_data = data_tensor.reshape(shot_num, opt.inputdata_size_C, opt.inputdata_size_H, opt.inputdata_size_W)

            optimizer_G.zero_grad()
            gen_data = vae.decode(denormalize_tensor(adaptor_g(z), max_val, min_val)).sample

            #Adversarial loss
            loss_G = -ms_ssim((gen_data+1)/2, (real_data+1)/2, data_range=1)

            ms_ssim_data[epoch] = ms_ssim_cal(gen_data, real_data)
            lpips_data[epoch] = lpips_cal(real_data, gen_data.clamp(-1.0,1.0), lpips_loss) 
            psnr_data[epoch] = psnr(real_data, gen_data)
            loss_G.backward()
            optimizer_G.step() #update the parameters of generator

            print (
                 "[Epoch %d/%d]  [adaptor_g loss: %f]"
                    %(epoch, opt.epochs_adaptor, loss_G.item())
                    )
        ms_ssim_opt[step] = ms_ssim_data[199]
        lpips_opt[step] = lpips_data[199] 
        psnr_opt[step] = psnr_data[199]   
    
 
    print( "[psnr: %f] [ms_ssim: %f]  [lpips: %f]"
          % (np.mean(psnr_ori), np.mean(ms_ssim_ori), np.mean(lpips_ori)) )
    print( "[psnr: %f] [ms_ssim: %f]  [lpips: %f]"
          % (np.mean(psnr_opt[psnr_opt>20]), np.mean(ms_ssim_opt[ms_ssim_opt>10]), np.mean(lpips_opt[lpips_opt<0.2])) )




    '''
    fig = plt.figure(figsize=(10,10), constrained_layout = True)
    gs = fig.add_gridspec(5,3)
    for i in range(5):
        f_ax = fig.add_subplot(gs[i,0])
        f_ax.imshow((((data_tensor_real[i].squeeze().permute(1,2,0)+1)*127.5).cpu().numpy()).astype(np.uint8))
        f_ax.axis("off")
        f_ax = fig.add_subplot(gs[i,1])
        f_ax.imshow((((data_tensor_ori[i].squeeze().permute(1,2,0)+1)*127.5).cpu().numpy()).astype(np.uint8))
        f_ax.axis("off")
        f_ax = fig.add_subplot(gs[i,2])
        f_ax.imshow((((data_tensor_adaptor[i].squeeze().permute(1,2,0)+1)*127.5).cpu().numpy()).astype(np.uint8))
        f_ax.axis("off")
   
    plt.savefig('d:\\Python\\SemCom_LDM\\saved_data\\test_adaptor_mnist.svg') 

    '''


        


        
        
 






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










    