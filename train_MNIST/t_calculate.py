import os
import torch

from torchvision import datasets, transforms

from model import Generator_DC,  CVAE_Encoder
from torch.utils.data import Dataset
from Config import dic_obj as opt
from matplotlib import pyplot as plt

from pytorch_msssim import ssim, ms_ssim
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
import argparse
import torch.nn.functional as F
import math
from JPEG2000_LDPC import JPEG2000_LDPC
import numpy as np
from diffusers import AutoencoderKL 

from consistency_model_training import UNet
from consistency_models import ConsistencySamplingAndEditing, karras_schedule
from utils import pad_dims_like

from consistency_models import model_forward_wrapper
import time

def round_sigma(sigma):
    return torch.as_tensor(sigma)
    
def t_seq(N):
    
    sigma_min =0.002
    sigma_max = 2
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
    ms_ssim_output = ms_ssim((x+1)/2, (y+1)/2, data_range=1)
    ms_ssim_output = -10*torch.log10(1-ms_ssim_output)
    return ms_ssim_output

def lpips_cal(x,y,lpips_loss):
    x = x.cuda()
    y =y.cuda()
    lpips_output = lpips_loss(x,y)
    #lpips_output = F.mse_loss(x,y)
    return lpips_output

import glob
from PIL import Image
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



    '''
    vae = ConvVAE().cuda()
    vae.load_state_dict(torch.load(f'd:\\Python\\SemCom\\saved_model\\CVAE_model_of_Dog.pth'))
    for param in vae.parameters():
        param.requires_grad = False 
    vae.eval()  
    '''

    '''
    vae = AutoencoderKL.from_pretrained("D:\\Python\\SemCom\\saved_model\\diffusion")
    vae = vae.to(device)
    for param in vae.parameters():
        param.requires_grad = False 
    vae.eval()  
    '''
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

    #load CD model
    CD_model = UNet.from_pretrained('D:\\Python\\SemCom\\saved_model\\new\\mnist_6').cuda()
    for param in CD_model.parameters():
        param.requires_grad = False 
    CD_model.eval()  

    consistency_sample = ConsistencySamplingAndEditing()


    #load VE based diffusion model random shedule, noise variance 


    #load dataset
    transform = transforms.Compose([transforms.Resize(opt.inputdata_size_H), transforms.ToTensor(), transforms.Normalize(mean=[0.5], std=[0.5])  ])
    #transform = transforms.Compose([transforms.Resize(opt.inputdata_size_H), transforms.CenterCrop(opt.inputdata_size_W), transforms.PILToTensor(), transforms.ConvertImageDtype(torch.float), transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])])    

    # use MNIST dataset
    train_dataset = datasets.MNIST('d:\\Python\\SemCom\\saved_data', train=True, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(train_dataset, batch_size = 1, shuffle=True)


    SNR_sum = np.array([0,2,4,6,8,10,12,14,16,18,20])
    t_VE = np.zeros((11,100))
    t_EECD = np.zeros((11,100))
    t_sequence =   karras_schedule(100, 0.002, 2, 7, 'cuda')
    print(t_sequence)
   

    for step, (input_data, labels) in enumerate(test_loader):


        input_data =input_data.cuda()
        start_time = time.time()
        z, mean_value, std_value = E(input_data)
        end_time = time.time()

        time_len = end_time - start_time

        t_VE[:,step] = t_VE[:,step] + time_len*0.5
        t_EECD[:,step] = t_EECD[:,step] + time_len*0.5

        z =z.reshape(z.shape[0],1,8,8)
        z,max_val,min_val = normalize_fun(z)
        ii = 0

        for SNR_num in SNR_sum:
            print(SNR_num)

            var_signal = torch.var(z)
            rate_value = 10**(SNR_num/10)
            sigma_noise = math.sqrt(var_signal/rate_value)

            #add noise
        
            closest_index = torch.argmin(torch.abs(t_sequence - sigma_noise)).cuda()
            print(sigma_noise)

            noise = torch.randn_like(z).cuda()  
        
            closest_value = t_sequence[closest_index].cuda()
            before_values = t_sequence[:closest_index+1].cuda()
            noisy_z = z + noise*closest_value

            flip_values = torch.flip(before_values[2:], [0])
            print(flip_values)
        

            # VE calculate
            z_VE = noisy_z.clone()
            start_time = time.time()
        
            for i in range(len(flip_values)):
                sigma = torch.full((noisy_z.shape[0],), flip_values[i], dtype=noisy_z.dtype, device=noisy_z.device)
                z_VE =model_forward_wrapper(CD_model, z_VE, sigma).clamp(min=-1.0, max=1.0)

                if i == (len(flip_values)-1):
                    break
                noise = torch.randn_like(z_VE).cuda()
                z_VE = z_VE + noise*flip_values[i+1]


            z_VE = denormalize_tensor(z_VE, max_val, min_val)
            x_VE = G(z_VE.reshape(z_VE.shape[0],64,1,1))

            end_time = time.time()

            time_len = end_time - start_time
            t_VE[ii, step] = t_VE[ii, step] + time_len


            # EECD calculate

            # 提取第一个元素
            first_element = flip_values[0]

            # 计算最中间元素的索引
            middle_index = len(flip_values) // 2

            # 提取最中间的元素
            middle_element = flip_values[middle_index]

            # 组合成新的张量
            new_tensor = torch.tensor([first_element, middle_element])
            print(new_tensor)

            z_CD = noisy_z.clone()
            start_time = time.time()
            sigma = torch.full((noisy_z.shape[0],), new_tensor[0], dtype=noisy_z.dtype, device=noisy_z.device)
            z_CD = model_forward_wrapper(CD_model, z_CD, sigma).clamp(min=-1.0, max=1.0)
            noise = torch.randn_like(z_CD).cuda()
            z_CD = z_CD + noise*new_tensor[1]
            sigma = torch.full((noisy_z.shape[0],), new_tensor[1], dtype=noisy_z.dtype, device=noisy_z.device)
            z_CD = model_forward_wrapper(CD_model, z_CD, sigma).clamp(min=-1.0, max=1.0)            

            z_CD = denormalize_tensor(z_CD, max_val, min_val)

            x_CD = G(z_CD.reshape(z_CD.shape[0],64,1,1))
            end_time = time.time()
            time_len = end_time - start_time
            t_EECD[ii, step] = t_EECD[ii, step] + time_len

            ii= ii + 1

        
        


        if step >= 99:
            break
    
    print(np.mean(t_VE, axis=1))
    print(np.mean(t_EECD, axis=1))

  





        




















  



    


