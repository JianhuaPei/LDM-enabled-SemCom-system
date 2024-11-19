import argparse
import glob
import os
import numpy as np
import pandas as pd
from torchvision import datasets, transforms
import torch
import lpips
#from model import  CVAE_Encoder

from Config import dic_obj as opt
from sklearn import preprocessing
from matplotlib import pyplot as plt

from PIL import Image
from torch.utils.data import Dataset
from pytorch_msssim import ssim, ms_ssim
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
import torch.nn.functional as F
from model import Generator_DC, Discriminator_DC, CVAE_Encoder

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
    
def lpips_cal(x,y,lpips_loss):
    x = x
    y =y

    return F.mse_loss(x, y)



def ms_ssim_cal(x,y):
    ms_ssim_output = ssim((x+1)/2, (y+1)/2, data_range=1)
    ms_ssim_output = -10*torch.log10(1-ms_ssim_output)
    return ms_ssim_output    

def psnr(x, y, max_val=1.0):
    mse = F.mse_loss(x, y)
    psnr = 10 * torch.log10((max_val**2) / mse)
    return psnr.item()


def loss_fn(recon_x, x, mu, logvar, critiron):   # defining loss function for va-AE (loss= reconstruction loss + KLD (to analyse if we have normal distributon))

    
    BCE = ssim((recon_x+1)/2, (x+1)/2, data_range=1)
    #BCE = MS_SSIM(recon_x,x)

    # source: Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # KLD is equal to 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    return -BCE + 0.01*KLD, BCE, KLD   




def norms(Z):
    # to compute norm
    return Z.view(Z.shape[0],-1).norm(dim=1)[:,None,None,None]

def Encode_output(E, real_data):
    z, mu, var = E(real_data)
    z = z.reshape(real_data.shape[0], opt.latent_dim, 1, 1)
    return z


def PGD(G, E, real_data, epsilon = 0.2, alpha = 0.2, num_iter=10):
    # initialize with zeros
    delta = torch.zeros_like(real_data, requires_grad = True).cuda()
    delta_1 =delta.detach()
    critiron = torch.nn.MSELoss()  
    fake_data = G(Encode_output(E,real_data))+ 0.00001
    epsilon = epsilon *opt.inputdata_size_H
    #MS_ssim = MS_SSIM()
    for i in range(num_iter):

        loss =  -ssim((G(Encode_output(E,real_data+delta))+1)/2, (fake_data+1)/2, data_range=1)
        
        #loss = -MS_ssim(real_data+delta, fake_data)
        loss.backward()
        if torch.isnan(loss).any():
            print("Loss contain NaN values")
            delta = delta_1
            break
        print(loss)
        delta_1 = delta.detach()
        #L_inifinity norm
        delta.data = (delta + alpha*delta.grad.detach().sign()).clamp(-epsilon,epsilon)
        #L_2 norm
        delta.data += alpha*delta.grad.detach() / norms(delta.grad.detach())
        delta.data = torch.min(torch.max(delta.detach(), -real_data), 1-real_data) # to clip the real_data+delta to [0,1]
        delta.data *= epsilon/norms(delta.detach()).clamp(min=epsilon)
        delta.grad.zero_()
    return  delta.detach()

def recons(VAE, data):
    z, mu, logvar= VAE.encoder(data.cuda())
    recons_data = VAE.decoder(z)
    #mean, logvar = VAE.encoder(data)
    #z = VAE.reparameterize(mean, logvar)
    #recons_data = VAE.decoder(z)
    return recons_data

def add_noise(x, snr):

    
    # 计算信号功率
    signal_power = torch.mean(x ** 2)
    
    # 将信噪比从dB转换为线性比例
    snr_linear = 10 ** (snr / 10)
    
    # 计算噪声功率
    noise_power = signal_power / snr_linear
    
    # 生成噪声
    noise = torch.randn_like(x) * torch.sqrt(noise_power)
    
    # 添加噪声到信号
    y = x + noise
    
    # 确保输出仍在 [-1, 1] 范围内
    y = torch.clamp(y, -1, 1)
    
    return y



if __name__ == '__main__':
    
    input_shape = ( opt.inputdata_size_C, opt.inputdata_size_H, opt.inputdata_size_W)
    device = "cuda"

    G=Generator_DC(input_dim = opt.latent_dim).cuda()
    G.load_state_dict(torch.load(f'd:\\Python\\SemCom\\saved_model\\generator_dc_model_of_MNIST.pth'))
    for param in G.parameters():
        param.requires_grad = False  
    G.eval()

    E=CVAE_Encoder().cuda()
    E.load_state_dict(torch.load(f'd:\\Python\\SemCom\\saved_model\\v_encoder_dc_model_of_MNIST.pth'))
    for param in E.parameters():
        param.requires_grad = False    
    E.eval()

    


    lpips_loss = LearnedPerceptualImagePatchSimilarity(net_type='vgg')
    lpips_loss = lpips_loss.to(device)
    for param in lpips_loss.parameters():
        param.requires_grad = False 
    lpips_loss.eval() 

    #load dataset
    #load dataset

    transform = transforms.Compose([transforms.Resize(opt.inputdata_size_H), transforms.ToTensor(), transforms.Normalize(mean=[0.5], std=[0.5])  ])
    #transform = transforms.Compose([transforms.Resize(opt.inputdata_size_H), transforms.CenterCrop(opt.inputdata_size_W), transforms.PILToTensor(), transforms.ConvertImageDtype(torch.float), transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])])    

    # use MNIST dataset
    train_dataset = datasets.MNIST('d:\\Python\\SemCom_LDM\\saved_data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = opt.batch_size_VAE, shuffle=True)


    # test on original encoder

    test_list = [1, 2, 3, 4, 5, 6]

    for test_num in test_list:

        test_delta = test_num * 0.05
        test_snr = test_num * 2.5 + 2.5

        for step, (input_data, labels) in enumerate(train_loader):
            input_data = input_data.cuda()

            delta = PGD(G, E, input_data, epsilon = test_delta )

            
            delta_data = input_data + delta
            noisy_data = add_noise(input_data, test_snr)

            z_delta, mean_val, std_val = E(delta_data)
            z_noisy, mean_val, std_val = E(noisy_data)

            delta_data_recon = G(z_delta.reshape(z_delta.shape[0], opt.latent_dim,1,1))
            noisy_data_recon = G(z_noisy.reshape(z_delta.shape[0], opt.latent_dim,1,1))
            print(test_num)
            print("[delta psnr: %f] [delta ssim: %f] [delta mse: %f]" %((psnr(input_data, delta_data_recon)), (ms_ssim_cal(delta_data_recon, input_data)), (lpips_cal(input_data, delta_data_recon, lpips_loss) )))
            print("[noise psnr: %f] [noise ssim: %f] [noise mse: %f]" %((psnr(input_data, noisy_data_recon)), (ms_ssim_cal(noisy_data_recon, input_data)), (lpips_cal(input_data, noisy_data_recon, lpips_loss) )))


            if step >= 0:
                break






    
    #save the PGD data for training
    for step, (input_data,labels) in enumerate(train_loader):
        input_data = input_data.cuda()

        print(step)
        delta = PGD(G, E, input_data )
        corrupted_data = input_data + delta

        #save input_data and corrupted_data
        data_path_input = 'd:\\Python\\SemCom\\saved_data\\MNIST\\PGD_data\\input_data_'+str(step)+'.pt'
        torch.save(input_data, data_path_input)

        data_path_corrupt = 'd:\\Python\\SemCom\\saved_data\\MNIST\\PGD_data\\corrupt_data_'+str(step)+'.pt'
        torch.save(corrupted_data, data_path_corrupt)



    
    model = CVAE_Encoder().cuda()
    critiron = torch.nn.MSELoss()
    optimizer  = torch.optim.Adam(model.parameters(), lr = opt.lr_VAE)

    iteration = 0
        
    for epoch in range(opt.epochs_VAE):
        for step, (input_data,labels) in enumerate(train_loader):
            input_data = input_data.cuda()

            optimizer.zero_grad()

            target = input_data.reshape(input_data.shape[0],opt.inputdata_size_C, opt.inputdata_size_H, opt.inputdata_size_W)
            target_emb, mu, logvar  = model(target)
            target_emb = target_emb.reshape(input_data.shape[0], opt.latent_dim, 1, 1)
            iteration = epoch*len(train_loader)+step
            
            gen_data = G(target_emb)
            loss, bce, kld = loss_fn(gen_data, target, mu, logvar, critiron )
            loss.backward()
            optimizer.step()
            

            data_path_input = 'd:\\Python\\SemCom\\saved_data\\MNIST\\PGD_data\\input_data_'+str(step)+'.pt'
            input_target = torch.load( data_path_input)

            data_path_corrupt = 'd:\\Python\\SemCom\\saved_data\\MNIST\\PGD_data\\corrupt_data_'+str(step)+'.pt'
            corrupt_target = torch.load(data_path_corrupt)

            target_emb_aug, mu, logvar = model(corrupt_target.cuda())
            target_emb_aug = target_emb_aug.reshape(input_data.shape[0], opt.latent_dim,1,1)
            gen_data_aug = G(target_emb_aug)
            loss, bce, kld = loss_fn(gen_data_aug, input_target.cuda(), mu, logvar, critiron )
            loss.backward()
            optimizer.step()
           
            
        print (
                "[Epoch %d/%d] [iteration %d] [Loss: %f] [BCE: %f] [KLD: %f]"
                %(epoch, opt.epochs_VAE, iteration, loss.item(), bce.item(), kld.item())
                )  
    
         
    #save encoder model 
    torch.save(model.state_dict(), 'd:\\Python\\SemCom\\saved_model\\robust_encoder_model_of_MNIST.pth')
    

    E_new=CVAE_Encoder().cuda()
    E_new.load_state_dict(torch.load(f'd:\\Python\\SemCom\\saved_model\\robust_encoder_model_of_MNIST.pth'))
    for param in E_new.parameters():
        param.requires_grad = False  
    E_new.eval()


    #test on new encoder

    test_list = [1, 2, 3, 4, 5, 6]

    for test_num in test_list:

        test_delta = test_num * 0.05
        test_snr = test_num * 2.5 + 2.5

        for step, (input_data,labels) in enumerate(train_loader):
            input_data = input_data.cuda()

            delta = PGD(G, E, input_data, epsilon = test_delta )

            

            delta_data = input_data + delta
            noisy_data = add_noise(input_data, test_snr)
            z_delta, mean_val, std_val = E_new(delta_data)
            z_noisy, mean_val, std_val = E_new(noisy_data)
            delta_data_recon = G(z_delta.reshape(z_delta.shape[0], opt.latent_dim,1,1))
            noisy_data_recon = G(z_noisy.reshape(z_delta.shape[0], opt.latent_dim,1,1))
            print(test_num)
            print("[delta psnr: %f] [delta ssim: %f] [delta mse: %f]" %((psnr(input_data, delta_data_recon)), (ms_ssim_cal(delta_data_recon, input_data)), (lpips_cal(input_data, delta_data_recon, lpips_loss) )))
            print("[noise psnr: %f] [noise ssim: %f] [noise mse: %f]" %((psnr(input_data, noisy_data_recon)), (ms_ssim_cal(noisy_data_recon, input_data)), (lpips_cal(input_data, noisy_data_recon, lpips_loss) )))


            if step >= 0:
                break
