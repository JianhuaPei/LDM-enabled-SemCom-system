import numpy as np
import torch
import torch.nn as nn
from Config import dic_obj as opt
import torch.nn.init as init
import math
import torch.nn.functional as F

input_shape=(opt.inputdata_size_C, opt.inputdata_size_H, opt.inputdata_size_W) #input image size H*W*C

class Flatten(nn.Module):   # falatten the image to vector (incloding the batch)
    def forward(self, input):
        return input.view(input.size(0), -1)   # (batch_size, channels,sz,sz) ---> (batch_size,channels*sz*sz)

class UnFlatten(nn.Module):  # reverse of flatten function
    def forward(self, input, size=2048):
        return input.view(input.size(0), size, 3, 3)  # (batch_size, channels*sz*sz) ---> (batch_size,channels,sz,sz)



'''
class Adaptor_g(nn.Module):
    def __init__(self):
        super(Adaptor_g, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(4, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(128, 4, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )  

    def forward(self, x):

        out = self.conv1(x) 
        out = self.conv2(out)
        return out
'''

def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()

class Adaptor_g(nn.Module):
    def __init__(self, input_size = 4*32*32, output_size = 4*32*32):
        super(Adaptor_g, self).__init__()
        self.adaptor =   nn.Sequential(
            nn.Linear(input_size,input_size),
            nn.ReLU(),
            nn.Linear(output_size,output_size),
            nn.Tanh()
        )          
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)   

    def forward(self, x):
        x = x.reshape(x.shape[0], 4*32*32)
        out = self.adaptor(x) 
        return out.reshape(x.shape[0], 4, 32, 32)





class CVAE_Encoder(nn.Module):
    def __init__(self, h_dim=2048*3*3, z_dim=opt.latent_dim):
        super(CVAE_Encoder, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU()
        )   
        self.conv3 = nn.Sequential(                
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU()
            )  # 128*32x32
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU()
            )  # 256*16x16
        self.conv5 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU()
            )
        self.conv6 = nn.Sequential(
            nn.Conv2d(1024, 2048, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU()
            )  # 1024*4x4 
     
        
        self.fl=Flatten()
        self.fc1 = nn.Linear(h_dim, z_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)



    def reparameterize(self, mu, logvar):  # producing latent layer (Guassian distribution )
        std = logvar.mul(0.5).exp_().cuda()       # hint: var=std^2
        # return torch.normal(mu, std)
        esp = torch.randn(*mu.size()).cuda()   # normal unit distribution in shape of mu
        z = mu + std * esp     # mu:mean  std: standard deviation
        return z

    def bottleneck(self, h):      # hidden layer ---> mean layer + logvar layer
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar 
        
    def forward(self, x):

        out = self.conv1(x) #32*128*128
        out = self.conv2(out) # 64*64*64
        out = self.conv3(out) # 128*32*32
        out = self.conv4(out) # 256*16*16
        out = self.conv5(out) # 512*8*8
        out = self.conv6(out) # 512*8*8
        h = self.fl(out) # flatten as 1024*4*4
        z, mu, logvar = self.bottleneck(h)
        return z, mu, logvar


class CVAE_Decoder(nn.Module):
    def __init__(self, h_dim=2048*3*3, z_dim=opt.latent_dim):
        super(CVAE_Decoder, self).__init__()
        self.fc3 = nn.Linear(z_dim, h_dim) # 1024*4*4
        self.unfl = UnFlatten()
        self.conv1 = nn.Sequential(nn.ConvTranspose2d(2048,1024,kernel_size=4,stride=2,padding=1),
            nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU()
        )  # 512*8x8        
        self.conv2 = nn.Sequential(nn.ConvTranspose2d(1024,512,kernel_size=4,stride=2,padding=1),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU()
        )  # 512*8x8
        self.conv3 = nn.Sequential(nn.ConvTranspose2d(512,256,kernel_size=4,stride=2,padding=1),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU()
        )  # 256*16x16
        self.conv4 = nn.Sequential(nn.ConvTranspose2d(256,128,kernel_size=4,stride=2,padding=1),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU()
        )  # 128*32x32
        self.conv5 = nn.Sequential(nn.ConvTranspose2d(128,64,kernel_size=4,stride=2,padding=1),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU()
        )  # 64*64x64
        self.conv6 = nn.Sequential(nn.ConvTranspose2d(64,3,kernel_size=4,stride=2,padding=1),
                                nn.Tanh()
        )  # 3x256*256 


    def forward(self, z):
        h = self.fc3(z) # 1024*4*4
        h = self.unfl(h) # 1024*4*4
        out = self.conv1(h) # 512*8*8
        out = self.conv2(out) # 256*16*16
        out = self.conv3(out) # 128*32*32
        out = self.conv4(out) # 64*64*64
        out = self.conv5(out) # 32*128*128
        out = self.conv6(out) # 32*128*128
        return out


class ConvVAE(nn.Module):
    def __init__(self):
        super(ConvVAE, self).__init__()
        self.encoder = CVAE_Encoder()
        
        self.decoder = CVAE_Decoder()

    def forward(self, x):
        z, mu, logvar = self.encoder(x)
        return self.decoder(z), mu, logvar 
