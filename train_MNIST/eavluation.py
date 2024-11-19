import os
import torch

from torchvision import datasets, transforms

from model import Generator_DC,  CVAE_Encoder
from torch.utils.data import Dataset
from Config import dic_obj as opt
from matplotlib import pyplot as plt
from networks import VEPrecond
from loss import VELoss
from diffusion import Diffusion
from pytorch_msssim import ssim, ms_ssim
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
import yaml
import argparse
import torch.nn.functional as F
import math
from JPEG2000_LDPC import JPEG2000_LDPC
import numpy as np
from consistency_model_training import UNet
from consistency_models import ConsistencySamplingAndEditing

def round_sigma(sigma):
    return torch.as_tensor(sigma)
    
def t_seq(N):
    
    sigma_min =0.002
    sigma_max =10
    rho=7
        
    # Time step discretization.
    step_indices = torch.arange(N, dtype=torch.float64) #, device=latents.device)
    t_steps = (sigma_min ** (1 / rho) + step_indices / (N - 1) * (sigma_max ** (1 / rho) -sigma_min ** (1 / rho))) ** rho
    t_steps = torch.cat([torch.zeros_like(t_steps[:1]), round_sigma(t_steps)])  
    return t_steps


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace

def psnr(x, y, max_val=1.0):
    mse = F.mse_loss(x, y)
    psnr = 10 * torch.log10((max_val**2) / mse)
    return psnr.item()

def ms_ssim_cal(x,y):
    ms_ssim_output = ssim((x+1)/2, (y+1)/2, data_range=1)
    ms_ssim_output = -10*torch.log10(1-ms_ssim_output)
    return ms_ssim_output

def lpips_cal(x,y,lpips_loss):
    #lpips_output = lpips_loss(x,y)
    lpips_output = F.mse_loss(x,y)
    return lpips_output

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
    x_denormalized = torch.zeros_like(x_normalized)
    
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

    #load VE based diffusion model random shedule, noise variance 



    #load CD model
    CD_model = UNet.from_pretrained('D:\\Python\\SemCom\\saved_model\\new').cuda()
    for param in CD_model.parameters():
        param.requires_grad = False 
    CD_model.eval()  

    consistency_sample = ConsistencySamplingAndEditing()
  

    transform = transforms.Compose([transforms.Resize(opt.inputdata_size_H), transforms.ToTensor(), transforms.Normalize(mean=[0.5], std=[0.5])  ])
    #transform = transforms.Compose([transforms.Resize(opt.inputdata_size_H), transforms.CenterCrop(opt.inputdata_size_W), transforms.PILToTensor(), transforms.ConvertImageDtype(torch.float), transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])])

    # use MNIST dataset
    dataset = datasets.MNIST('d:\\Python\\SemCom\\saved_data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(dataset, batch_size = opt.batch_size_diffusion, shuffle=True)

    lpips_loss = LearnedPerceptualImagePatchSimilarity(net_type='vgg')

    SNR_num = 10
    t_sequence = t_seq(N=int(np.ceil(np.sqrt( ((150 + 1)**2 - 2**2) + 2**2) - 1) + 1))
    PSNR_avg_JSCC = 0
    ms_ssim_avg_JSCC = 0
    lpips_avg_JSCC = 0
    PSNR_avg_VE = 0
    ms_ssim_avg_VE = 0
    lpips_avg_VE = 0    
    PSNR_avg_CD = 0
    ms_ssim_avg_CD = 0
    lpips_avg_CD = 0
    PSNR_avg_LDPC = 0
    ms_ssim_avg_LDPC = 0
    lpips_avg_LDPC = 0    

    for step, (input_data, labels) in enumerate(test_loader):


        input_data =input_data.cuda()

        z,mu,var = E(input_data)
        z =z.reshape(z.shape[0],1,8,8)
        z,max_val,min_val = normalize_fun(z)


        var_signal = torch.var(z)
        rate_value = 10**(SNR_num/10)
        sigma_noise = math.sqrt(var_signal/rate_value)

        #add noise
        
        noise = torch.randn_like(z) * sigma_noise
        noisy_z = z + noise.cuda() 

        closest_index = torch.argmin(torch.abs(t_sequence - sigma_noise))
        print(sigma_noise)

        closest_value = t_sequence[closest_index].cuda()
        before_values = t_sequence[:closest_index+1].cuda()
        filp_values = torch.flip(before_values, dims=[0])
        print(closest_value)

        with torch.no_grad():
            z_CD = consistency_sample(CD_model, noisy_z, sigmas = [closest_value], clip_denoised=True, verbose = True) #1-step
            z_VE = consistency_sample(CD_model, noisy_z, sigmas = filp_values, clip_denoised=True, verbose = True) #m-step

        z_VE = denormalize_tensor(z_VE, max_val, min_val)
        z_CD = denormalize_tensor(z_CD, max_val, min_val)
        noisy_z = denormalize_tensor(noisy_z, max_val, min_val)
        x_JSCC = G(noisy_z.reshape(noisy_z.shape[0], opt.latent_dim,1,1))
        x_VE = G(z_VE.reshape(z_VE.shape[0],opt.latent_dim,1,1))
        x_CD = G(z_CD.reshape(z_CD.shape[0],opt.latent_dim,1,1))
        x_LDPC = JPEG2000_LDPC(input_data, compression_ratio =100, snr_num = SNR_num)
        #x_LDPC = input_data

        #calculate PSNR, MS-SSIM, LPIPS
        PSNR_avg_JSCC = PSNR_avg_JSCC + psnr(input_data, x_JSCC)
        ms_ssim_avg_JSCC = ms_ssim_avg_JSCC + ms_ssim_cal(input_data,x_JSCC)
        lpips_avg_JSCC = lpips_avg_JSCC + lpips_cal(input_data,x_JSCC,lpips_loss)
        PSNR_avg_VE = PSNR_avg_VE + psnr(input_data, x_VE)
        ms_ssim_avg_VE = ms_ssim_avg_VE + ms_ssim_cal(input_data,x_VE)
        lpips_avg_VE = lpips_avg_VE + lpips_cal(input_data,x_VE,lpips_loss)
        PSNR_avg_CD = PSNR_avg_CD + psnr(input_data, x_CD)
        ms_ssim_avg_CD = ms_ssim_avg_CD + ms_ssim_cal(input_data,x_CD)
        lpips_avg_CD = lpips_avg_CD + lpips_cal(input_data,x_CD,lpips_loss)
        PSNR_avg_LDPC = PSNR_avg_LDPC + psnr(input_data, x_LDPC)
        ms_ssim_avg_LDPC = ms_ssim_avg_LDPC + ms_ssim_cal(input_data,x_LDPC)
        lpips_avg_LDPC = lpips_avg_LDPC + lpips_cal(input_data,x_LDPC,lpips_loss)    


        if step > 1:
            break   

    PSNR_avg_JSCC = PSNR_avg_JSCC/(step+1)
    ms_ssim_avg_JSCC = ms_ssim_avg_JSCC /(step+1)
    lpips_avg_JSCC = lpips_avg_JSCC /(step+1)
    PSNR_avg_VE = PSNR_avg_VE /(step+1)
    ms_ssim_avg_VE = ms_ssim_avg_VE /(step+1)
    lpips_avg_VE = lpips_avg_VE /(step+1)
    PSNR_avg_CD = PSNR_avg_CD /(step+1)
    ms_ssim_avg_CD = ms_ssim_avg_CD /(step+1)
    lpips_avg_CD = lpips_avg_CD  /(step+1)
    PSNR_avg_LDPC = PSNR_avg_LDPC /(step+1)
    ms_ssim_avg_LDPC = ms_ssim_avg_LDPC /(step+1)
    lpips_avg_LDPC = lpips_avg_LDPC  /(step+1)

    print (
            "[LDPC_PSNR %f] [LDPC_MS_SSIM %f] [LDPC_LPIPS: %f] "
                %(PSNR_avg_LDPC, ms_ssim_avg_LDPC, lpips_avg_LDPC)
                )     

    print (
            "[JSCC_PSNR %f] [JSCC_MS_SSIM %f] [JSCC_LPIPS: %f] "
                %(PSNR_avg_JSCC, ms_ssim_avg_JSCC, lpips_avg_JSCC)
                )      
    print (
            "[VE_PSNR %f] [VE_MS_SSIM %f] [VE_LPIPS: %f] "
                %(PSNR_avg_VE, ms_ssim_avg_VE, lpips_avg_VE)
                )      
    print (
            "[CD_PSNR %f] [CD_MS_SSIM %f] [CD_LPIPS: %f] "
                %(PSNR_avg_CD, ms_ssim_avg_CD, lpips_avg_CD)
                )    
    
    #save images
    fig = plt.figure(figsize=(10,10), constrained_layout = True)
    gs = fig.add_gridspec(5,5)
    for i in range(5):
        f_ax = fig.add_subplot(gs[i,0])
        f_ax.imshow(input_data[i][0].detach().cpu().numpy(), cmap="gray")
        f_ax.axis("off")
        f_ax = fig.add_subplot(gs[i,1])
        f_ax.imshow(x_LDPC[i][0].detach().cpu().numpy(), cmap="gray")
        f_ax.axis("off")
        f_ax = fig.add_subplot(gs[i,2])
        f_ax.imshow(x_JSCC[i][0].detach().cpu().numpy(), cmap="gray")
        f_ax.axis("off")
        f_ax = fig.add_subplot(gs[i,3])
        f_ax.imshow(x_VE[i][0].detach().cpu().numpy(), cmap="gray")
        f_ax.axis("off")
        f_ax = fig.add_subplot(gs[i,4])
        f_ax.imshow(x_CD[i][0].detach().cpu().numpy(), cmap="gray")
        f_ax.axis("off")   
    plt.savefig('d:\\Python\\SemCom\\saved_data\\channel_AWGN_mnist.svg')      





        




















  



    


