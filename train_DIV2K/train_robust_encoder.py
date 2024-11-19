import argparse
import glob
import os
import numpy as np
import pandas as pd
from torchvision import datasets, transforms
import torch
import lpips
from model_new import ConvVAE, CVAE_Encoder
from Config import dic_obj as opt
from sklearn import preprocessing
from matplotlib import pyplot as plt
from PIL import Image
from torch.utils.data import Dataset
from pytorch_pretrained_gans import make_gan
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



def norms(Z):
    # to compute norm
    return Z.view(Z.shape[0],-1).norm(dim=1)[:,None,None,None]

def loss_fn(recon_x, x, mu, logvar, critiron):   # defining loss function for va-AE (loss= reconstruction loss + KLD (to analyse if we have normal distributon))

    
    BCE = ms_ssim((recon_x+1)/2, (x+1)/2, data_range=1)

    # source: Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # KLD is equal to 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    return -BCE + 0.01*KLD, BCE, KLD    



def PGD(VAE, real_data, epsilon = 0.07813, alpha = 0.4, num_iter=10):
    # initialize with zeros
    delta = torch.zeros_like(real_data, requires_grad = True).cuda()
    critiron = torch.nn.MSELoss()  
    diff = 0.00001
    difference = diff.cuda()
    fake_data = recons(VAE, real_data) + difference
    epsilon =(epsilon *opt.inputdata_size_H).cuda()
    alpha = alpha.cuda()
    
    for i in range(num_iter):

        loss =  -ms_ssim((recons(VAE, real_data+delta)+1)/2, (fake_data+1)/2, data_range=1)
        loss.backward()
        print(loss)
        
        #L_inifinity norm
        delta.data = (delta + alpha*delta.grad.detach().sign()).clamp(-epsilon,epsilon)
        #L_2 norm
        delta.data += alpha*delta.grad.detach() / norms(delta.grad.detach())
        delta.data = torch.min(torch.max(delta.detach(), -real_data), 1-real_data) # to clip the real_data+delta to [0,1]
        delta.data *= epsilon/norms(delta.detach()).clamp(min=epsilon)
        delta.grad.zero_()
        

    return delta.detach()

def recons(VAE, data):
    z, mu, logvar= VAE.encoder(data)
    recons_data = VAE.decoder(z)
    #mean, logvar = VAE.encoder(data)
    #z = VAE.reparameterize(mean, logvar)
    #recons_data = VAE.decoder(z)
    return recons_data

def plot_images_RBG(real_data, VAE, M):
    fig = plt.figure(figsize=(10,10), constrained_layout = True)
    gs = fig.add_gridspec(M,5)
    # M -> column data number N -> row data type
    delta = PGD(VAE,real_data)
    corrupted_data = real_data + delta
    #z = torch.tensor(np.random.normal(0,1,(real_data.shape[0], opt.latent_dim))).float()
    recons_data_original = recons(VAE, real_data)
    #recons_data_original = G(z)
    recons_data = recons(VAE, corrupted_data)
    for i in range(M):
        f_ax = fig.add_subplot(gs[i,0])
        f_ax.imshow((((real_data[i].squeeze().permute(1,2,0)+1)*127.5).cpu().numpy()).astype(np.uint8))
        f_ax.axis("off")
        f_ax = fig.add_subplot(gs[i,1])
        f_ax.imshow((((recons_data_original[i].squeeze().permute(1,2,0)+1)*127.5).cpu().numpy()).astype(np.uint8))
        f_ax.axis("off")
        f_ax = fig.add_subplot(gs[i,2])
        f_ax.imshow((((delta[i].squeeze().permute(1,2,0)+1)*127.5).cpu().numpy()).astype(np.uint8))
        f_ax.axis("off")
        f_ax = fig.add_subplot(gs[i,3])
        f_ax.imshow((((corrupted_data[i].squeeze().permute(1,2,0)+1)*127.5).cpu().numpy()).astype(np.uint8))
        f_ax.axis("off")
        f_ax = fig.add_subplot(gs[i,4])
        f_ax.imshow((((recons_data[i].squeeze().permute(1,2,0)+1)*127.5).cpu().numpy()).astype(np.uint8))
        f_ax.axis("off")
    plt.savefig('d:\\Python\\SemCom_LDM\\saved_data\\test_DIV2K_512.svg')


def plot_images_update(real_data, VAE, E_new, M):
    fig = plt.figure(figsize=(10,10), constrained_layout = True)
    gs = fig.add_gridspec(M,6)
    # M -> column data number N -> row data type
    delta = PGD(VAE,real_data)
    corrupted_data = real_data + delta
    #z = torch.tensor(np.random.normal(0,1,(real_data.shape[0], opt.latent_dim))).float()
    #z = z.reshape(real_data.shape[0], opt.latent_dim,1,1) 
    #recons_data_original = G(E(real_data).reshape(real_data.shape[0], opt.latent_dim, 1, 1))
    #recons_data_original = G(z)
    recons_data_original = recons(VAE, real_data)
    recons_data = recons(VAE, corrupted_data)
    recons_data_new = VAE.decoder(E_new(corrupted_data))
    for i in range(M):
        f_ax = fig.add_subplot(gs[i,0])
        f_ax.imshow((((real_data[i].squeeze().permute(1,2,0)+1)*127.5).cpu().numpy()).astype(np.uint8))
        f_ax.axis("off")
        f_ax = fig.add_subplot(gs[i,1])
        f_ax.imshow((((recons_data_original[i].squeeze().permute(1,2,0)+1)*127.5).cpu().numpy()).astype(np.uint8))
        f_ax.axis("off")
        f_ax = fig.add_subplot(gs[i,2])
        f_ax.imshow((((delta[i].squeeze().permute(1,2,0)+1)*127.5).cpu().numpy()).astype(np.uint8))
        f_ax.axis("off")
        f_ax = fig.add_subplot(gs[i,3])
        f_ax.imshow((((corrupted_data[i].squeeze().permute(1,2,0)+1)*127.5).cpu().numpy()).astype(np.uint8))
        f_ax.axis("off")
        f_ax = fig.add_subplot(gs[i,4])
        f_ax.imshow((((recons_data[i].squeeze().permute(1,2,0)+1)*127.5).cpu().numpy()).astype(np.uint8))
        f_ax.axis("off")   
        f_ax = fig.add_subplot(gs[i,5])
        f_ax.imshow((((recons_data_new[i].squeeze().permute(1,2,0)+1)*127.5).cpu().numpy()).astype(np.uint8))
        f_ax.axis("off")    
    plt.savefig('d:\\Python\\SemCom_LDM\\saved_data\\test_DIV2K_512_robust.svg') 

if __name__ == '__main__':
    
    input_shape = ( opt.inputdata_size_C, opt.inputdata_size_H, opt.inputdata_size_W)
    cuda = True if torch.cuda.is_available() else False
    device = "cuda"
    '''
    G = make_gan(gan_type='biggan',model_name='biggan-deep-256')
    G = G.to(device)
    for param in G.parameters():
        param.requires_grad = False
    G.eval()  
    '''

    VAE = ConvVAE().cuda()
    VAE.load_state_dict(torch.load(f'd:\\Python\\SemCom_LDM\\saved_model\\CVAE_model_of_DIV2K.pth'))
    for param in VAE.parameters():
        param.requires_grad = False   

    #load dataset
    data_path = 'D:\\Python\\SemCom_LDM\\saved_data\\DIV2K\\DIV2K_valid_HR\\DIV2K_valid_HR\\'

    transforms_train = transforms.Compose([transforms.Resize(opt.inputdata_size_H), transforms.CenterCrop(opt.inputdata_size_W), transforms.PILToTensor(), transforms.ConvertImageDtype(torch.float), transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])]) 
    my_dataset = Dataset_m(f'{data_path}', transform = transforms_train)
    train_loader = torch.utils.data.DataLoader(
        dataset = my_dataset,
        batch_size=opt.batch_size_VAE,
        shuffle=True,
    )  

    # test the delta
    for test_data in train_loader:
        test_data = test_data.to(device)
        break
    VAE.eval()

    plot_images_RBG(test_data, VAE, 5 )



    #save the PGD data for training
    for step, input_data in enumerate(train_loader):

        print(step)
        delta = PGD(VAE, input_data )
        corrupted_data = input_data + delta

        #save input_data and corrupted_data
        data_path_input = 'd:\\Python\\SemCom_LDM\\saved_data\\DIV2K\\PGD_data\\input_data_'+str(step)+'.pt'
        torch.save(input_data, data_path_input)

        data_path_corrupt = 'd:\\Python\\SemCom_LDM\\saved_data\\DIV2K\\PGD_data\\corrupt_data_'+str(step)+'.pt'
        torch.save(corrupted_data, data_path_corrupt)




    model = CVAE_Encoder()
    critiron = torch.nn.MSELoss()
    optimizer  = torch.optim.Adam(model.parameters(), lr = opt.lr_VAE)

    iteration = 0
        
    for epoch in range(opt.epochs_VAE):
        for step, input_data in enumerate(train_loader):

            optimizer.zero_grad()

            target = input_data.reshape(input_data.shape[0],opt.inputdata_size_C, opt.inputdata_size_H, opt.inputdata_size_W)
            target_emb, mu, logvar  = model(target)
            target_emb = target_emb.reshape(input_data.shape[0], opt.latent_dim, 1, 1)
            iteration = epoch*len(train_loader)+step
            
            gen_data = VAE.decoder(target_emb)
            loss, bce, kld = loss_fn(gen_data, target, mu, logvar, critiron )
            loss.backward()
            optimizer.step()
            

            data_path_input = 'd:\\Python\\SemCom_LDM\\saved_data\\DIV2K\\PGD_data\\input_data_'+str(step)+'.pt'
            input_target = torch.load( data_path_input)

            data_path_corrupt = 'd:\\Python\\SemCom_LDM\\saved_data\\DIV2K\\PGD_data\\corrupt_data_'+str(step)+'.pt'
            corrupt_target = torch.load(data_path_corrupt)

            target_emb_aug = model(corrupt_target)
            target_emb_aug = target_emb_aug.reshape(input_data.shape[0], opt.latent_dim)
            gen_data_aug = VAE.decoder(target_emb_aug)
            loss, bce, kld = loss_fn(gen_data, target, mu, logvar, critiron )
            loss.backward()
            optimizer.step()
           
            
        print (
                "[Epoch %d/%d] [iteration %d] [Loss: %f] [BCE: %f] [KLD: %f]"
                %(epoch, opt.epochs_VAE, iteration, loss.item(), bce.item(), kld.item())
                )  

         
    #save encoder model 
    torch.save(model.state_dict(), 'd:\\Python\\SemCom_LDM\\saved_model\\robust_encoder_model_of_DIV2K.pth')

    E_new=CVAE_Encoder()
    E_new.load_state_dict(torch.load(f'd:\\Python\\SemCom_LDM\\saved_model\\robust_encoder_model_of_DIV2K.pth'))
    for param in E_new.parameters():
        param.requires_grad = False  
    E_new.eval()
    plot_images_update(test_data, VAE, E_new, 5)
    
    