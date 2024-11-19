import numpy as np
import torch
import torch.nn as nn
from Config import dic_obj as opt
import torch.nn.init as init
import math

input_shape=(opt.inputdata_size_C, opt.inputdata_size_H, opt.inputdata_size_W) #input image size H*W*C

def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()



class Generator_DC(nn.Module):
    def __init__(self, input_dim):
        super(Generator_DC, self).__init__()

        self.main_module = nn.Sequential(
            nn.ConvTranspose2d(in_channels = input_dim, out_channels = 1024, kernel_size = 4, stride = 1, padding = 0),
            nn.BatchNorm2d(num_features = 1024),
            nn.ReLU(True),
            #4*4
            nn.ConvTranspose2d(in_channels = 1024, out_channels = 512, kernel_size = 4, stride = 2, padding = 1),
            nn.BatchNorm2d(num_features = 512),
            nn.ReLU(True),
            #8*8
            nn.ConvTranspose2d(in_channels = 512, out_channels = 256, kernel_size = 4, stride = 2, padding = 1),
            nn.BatchNorm2d(num_features = 256),
            nn.ReLU(True),
            #16*16
            nn.ConvTranspose2d(in_channels=256, out_channels= 128, kernel_size=4,stride=2, padding=1),
            nn.BatchNorm2d(num_features = 128),
            nn.ReLU(True),
            #32*32
            nn.ConvTranspose2d(in_channels=128, out_channels= 64, kernel_size=4,stride=2, padding=1),
            nn.BatchNorm2d(num_features = 64),
            nn.ReLU(True),          
            #64*64
            nn.ConvTranspose2d(in_channels=64, out_channels= 32, kernel_size=4,stride=2, padding=1),
            nn.BatchNorm2d(num_features = 32),
            nn.ReLU(True),             
            #128*128
            nn.ConvTranspose2d(in_channels=32, out_channels= opt.inputdata_size_C, kernel_size=4,stride=2, padding=1)
              

        )

        self.output = nn.Tanh()

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    def forward(self, z):
        x = self.main_module(z)
        x = self.output(x)
        return x
    

    



class Generator(nn.Module):
    def __init__(self,input_dim):
        super(Generator,self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers=[nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat,0.8))
            layers.append(nn.LeakyReLU(0.2,inplace=True))
            return layers

        self.model=nn.Sequential(
            *block(input_dim,128,normalize=False),
            *block(128,256),
            *block(256,512),
            *block(512,1024),
            *block(1024, 2048),
            *block(2048, 4096),
            nn.Linear(4096, int(np.prod(input_shape))),
            nn.Tanh()
            )

    def forward(self,z):
        inputdata = self.model(z)
        inputdata = inputdata.view(inputdata.shape[0],*input_shape)
        return inputdata

class Discriminator_DC(nn.Module):
    def __init__(self):
        super(Discriminator_DC, self).__init__()

        self.main_module = nn.Sequential(
            # 256*256
            nn.Conv2d(in_channels = opt.inputdata_size_C, out_channels = 32, kernel_size = 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace = True),
            #128*128
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(64, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            #64*64
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            #32*32
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(256, affine=True),
            nn.LeakyReLU(0.2, inplace=True),    
            #16*16
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(512, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            #8*8
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(1024, affine=True),
            nn.LeakyReLU(0.2, inplace=True)                                
        )
        self.output = nn.Sequential(
            nn.Conv2d(in_channels = 1024, out_channels=1, kernel_size=4, stride=1, padding=0),
            #nn.Sigmoid()
        )

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)    
    
    def forward(self, x):
        #x = x.view(x.shape[0],-1)
        x = self.main_module(x)
        return self.output(x)
    
    def feature_extraction(self, x):
        x = self.main_module(x)
        return x.view(-1, 1024*4*4)



class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model=nn.Sequential(
            nn.Linear(int(np.prod(input_shape)),512),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Linear(512,1024),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Linear(1024,2048),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Linear(2048,256),
            nn.LeakyReLU(0.2,inplace=True),
             
            )
        self.fc = nn.Linear(256,1)   #improvement 1 of WGAN: remove the sigmoid in the last layer of Discriminator
    def forward(self,inputdata):
        inputdata_flat = inputdata.view(inputdata.shape[0],-1)
        inputdata_flat = inputdata_flat.type(torch.float32)
        inputdata_flat = self.model(inputdata_flat)
        validity = self.fc(inputdata_flat)
        return validity

class Adaptor_g(nn.Module):
    def __init__(self, input_size = opt.latent_dim, output_size = opt.latent_dim):
        super(Adaptor_g, self).__init__()

        self.adaptor = nn.Sequential(
            nn.Linear(input_size,output_size),
            nn.ReLU()

        )

    def forward(self, inputdata, input_size = opt.latent_dim, output_size = opt.latent_dim):
        #flatten
        inputdata_flat = inputdata.view(inputdata.shape[0], input_size)
        #adaptor
        outputdata = self.adaptor(inputdata_flat)
        #reshape
        outputdata = outputdata.view(inputdata.shape[0], 1, output_size)        
        
        return outputdata

'''
class adaptor_d(nn.Module):
    def __init__(self, input_size = 256, output_size = 1):
        super(adaptor_d, self).__init__()

        self.adaptor = nn.Sequential(
            nn.Linear(input_size, output_size)

        )

    def forward(self, inputdata):
        inputdata_flat = inputdata.view(inputdata.shape[0],-1)
        inputdata_flat = inputdata_flat.type(torch.float32)
        validity = self.adaptor(inputdata_flat)
        return validity
'''
    



class AE(nn.Module):
    def __init__(self, input_size = opt.inputdata_size_H*opt.inputdata_size_W*opt.inputdata_size_C, output_size = opt.latent_dim):
        super(AE,self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Linear(512,256),
            nn.ReLU(),
            nn.Linear(256, output_size),
            nn.ReLU()
            
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(output_size, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, input_size),
            nn.Sigmoid(),
        )
        
    def forward(self, inputdata, input_size = opt.inputdata_size_H*opt.inputdata_size_W*opt.inputdata_size_C):
        
        """
        :param inputdata: [batch_size, 1, opt.inputdata_size_M, opt.inputdata_size_T]
        :return:
        """
        #flatten
        inputdata_flat = inputdata.view(inputdata.shape[0], input_size)
        #encoder
        inputdata_flat = self.encoder(inputdata_flat)
        #decoder
        inputdata_flat = self.decoder(inputdata_flat)
        #reshape
        outputdata = inputdata_flat.view(inputdata.shape[0], opt.inputdata_size_C, opt.inputdata_size_H, opt.inputdata_size_W)
        
        return outputdata
    
    
class Encoder(nn.Module):
    def __init__(self, input_size = opt.inputdata_size_H*opt.inputdata_size_W*opt.inputdata_size_C, output_size = opt.latent_dim):
        super(Encoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.ReLU(),
            nn.Linear(1024,512),
            nn.ReLU(),
            nn.Linear(512, output_size),
            nn.Tanh()    
        )
        
    def forward(self, inputdata, input_size = opt.inputdata_size_H*opt.inputdata_size_W*opt.inputdata_size_C, output_size = opt.latent_dim):
        #flatten
        inputdata_flat = inputdata.view(inputdata.shape[0], input_size)
        #encoder
        outputdata = self.encoder(inputdata_flat)
        #reshape
        outputdata = outputdata.view(inputdata.shape[0], 1, output_size)        
        
        
        return outputdata
    
    
class VAE(nn.Module):
    def __init__(self, input_size = opt.inputdata_size_H*opt.inputdata_size_W*opt.inputdata_size_C, output_size = opt.latent_dim):
        super(VAE, self).__init__()
        
        #[batch_size,1,M,T] => [batch_size, output_size]
        
        #u: [batch_size, output_size/2]
        #sigma: [batch_size, output_size/2]
        
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Linear(512,256),
            nn.ReLU(),
            nn.Linear(256, output_size),
            nn.ReLU()
            
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(int(output_size/2), 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, input_size),
            nn.Sigmoid(),
        )
            
    def forward(self, inputdata, input_size = opt.inputdata_size_H*opt.inputdata_size_W*opt.inputdata_size_C):
        """
        :param x: [batch_size, 1, M, T]
        :return:
        """
        
        #flatten
        inputdata_flat = inputdata.view(inputdata.shape[0], input_size)
        #encoder
        h_ = self.encoder(inputdata_flat)
        
        mu,sigma = h_.chunk(2, dim = 1)
        h = mu + sigma * torch.randn_like(sigma)
        
        kld = 0.5 * torch.sum( torch.pow(mu,2) + torch.pow(sigma, 2) - torch.log(1e-8 +torch.pow(sigma,2)) -1 )/(inputdata.shape[0]*input_size)
        
        #decoder
        x = self.decoder(h)
        
        #reshape
        x = x.view(inputdata.shape[0], opt.inputdata_size_C, opt.inputdata_size_H, opt.inputdata_size_W)
        
        return x, kld
    


class V_Encoder(nn.Module):
    def __init__(self, input_size = opt.inputdata_size_H*opt.inputdata_size_W*opt.inputdata_size_C, output_size = opt.latent_dim*2):
        super(V_Encoder, self).__init__()
        
        #[batch_size,1,M,T] => [batch_size, output_size]
        
        #u: [batch_size, output_size/2]
        #sigma: [batch_size, output_size/2]
        
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024,512),
            nn.ReLU(),
            nn.Linear(512,256),
            nn.ReLU(),
            nn.Linear(256, output_size),
            nn.ReLU()
            
        )
        
          
    def forward(self, inputdata, input_size = opt.inputdata_size_H*opt.inputdata_size_W*opt.inputdata_size_C, z_size = opt.latent_dim):
        """
        :param x: [batch_size, 1, M, T]
        :return:
        """
        
        #flatten
        inputdata_flat = inputdata.view(inputdata.shape[0], input_size)
        #encoder
        h_ = self.encoder(inputdata_flat)
        
        mu,sigma = h_.chunk(2, dim = 1)
        z = mu + sigma * torch.randn_like(sigma) #reparameterization
        
        #kld = 0.5 * torch.sum( torch.pow(mu,2) + torch.pow(sigma, 2) - torch.log(1e-8 +torch.pow(sigma,2)) -1 )/(inputdata.shape[0]*input_size)
        
        return z.view(inputdata.shape[0], 1, z_size) 
        
class Generator_256(nn.Module):
    def __init__(self, input_dim, M=4):
        super().__init__()
        self.M = M
        self.linear = nn.Linear(input_dim, M * M * 1024)
        self.main = nn.Sequential(
            nn.BatchNorm2d(1024),
            nn.ReLU(True),
            nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Conv2d(16, 3, kernel_size=3, stride=1, padding=1),
            nn.Tanh())
        self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                init.normal_(m.weight, std=0.02)
                init.zeros_(m.bias)

    def forward(self, z, *args, **kwargs):
        x = self.linear(z)
        x = x.view(x.size(0), -1, self.M, self.M)
        x = self.main(x)
        return x


class Discriminator_256(nn.Module):
    def __init__(self, M = 256):
        super().__init__()
        self.M = M

        self.main = nn.Sequential(
            # M
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            # M / 2
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            # M / 4
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            # M / 8
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            # M / 16
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            # M / 32
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(512, 1024, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True),  
            # M / 64
            nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1, inplace=True)

        )     

        self.linear = nn.Linear(4 * 4 * 1024, 1)
        self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                init.normal_(m.weight, std=0.02)
                init.zeros_(m.bias)

    def forward(self, x, *args, **kwargs):
        x = self.main(x)
        x = torch.flatten(x, start_dim=1)
        x = self.linear(x)
        return x


class Generator64(Generator_256):
    def __init__(self, input_dim):
        super().__init__(input_dim, M=4)


class Discriminator64(Discriminator_256):
    def __init__(self):
        super().__init__(M=256)            
        
        