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

            nn.ConvTranspose2d(in_channels = 1024, out_channels = 512, kernel_size = 4, stride = 2, padding = 1),
            nn.BatchNorm2d(num_features = 512),
            nn.ReLU(True),

            nn.ConvTranspose2d(in_channels = 512, out_channels = 256, kernel_size = 4, stride = 2, padding = 1),
            nn.BatchNorm2d(num_features = 256),
            nn.ReLU(True),

            nn.ConvTranspose2d(in_channels=256, out_channels= opt.inputdata_size_C, kernel_size=4,stride=2, padding=1)

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
            nn.Conv2d(in_channels = opt.inputdata_size_C, out_channels = 256, kernel_size = 4, stride=2, padding=1),
            nn.InstanceNorm2d(256, affine=True),
            nn.LeakyReLU(0.2, inplace = True),

            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(512, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

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
            nn.Linear(input_size,256),
            nn.ReLU(),
            nn.Linear(256,512),
            nn.ReLU(),
            nn.Linear(512,256),
            nn.ReLU(),
            nn.Linear(256,output_size),
            nn.Tanh()

        )

    def forward(self, inputdata, input_size = opt.latent_dim, output_size = opt.latent_dim):
        #flatten
        inputdata_flat = inputdata.view(inputdata.shape[0], input_size)
        #adaptor
        outputdata = self.adaptor(inputdata_flat)
        #reshape
        outputdata = outputdata.view(inputdata.shape[0], 1, output_size)        
        
        return outputdata
    
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)  

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
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Linear(512,256),
            nn.ReLU(),
            nn.Linear(256, output_size),
            nn.ReLU()    
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
            nn.Linear(input_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
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
            nn.Linear(input_size, 1024),
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
        
class Generator_32(nn.Module):
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
            nn.Conv2d(128, 3, kernel_size=3, stride=1, padding=1),
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


class Discriminator_32(nn.Module):
    def __init__(self, M = 32):
        super().__init__()
        self.M = M

        self.main = nn.Sequential(
            # M
            nn.Conv2d(3, 128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            # M / 2
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            # M / 4
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(512, 1024, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True),  
            # M / 8
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


class Generator32(Generator_32):
    def __init__(self, input_dim):
        super().__init__(input_dim, M=4)


class Discriminator32(Discriminator_32):
    def __init__(self):
        super().__init__(M=32)


class Flatten(nn.Module):   # falatten the image to vector (incloding the batch)
    def forward(self, input):
        return input.view(input.size(0), -1)   # (batch_size, channels,sz,sz) ---> (batch_size,channels*sz*sz)

class UnFlatten(nn.Module):  # reverse of flatten function
    def forward(self, input, size=256):
        return input.view(input.size(0), size, 2, 2)  # (batch_size, channels*sz*sz) ---> (batch_size,channels,sz,sz)


class CVAE_Encoder(nn.Module):
    def __init__(self, h_dim=1024*1*1, z_dim=opt.latent_dim):
        super(CVAE_Encoder, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True),
            nn.LeakyReLU(0.2, inplace=True)
            )  # 32*128*128

        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), 
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True),
            nn.LeakyReLU(0.2, inplace=True)
            ) # 64*64x64
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True),
            nn.LeakyReLU(0.2, inplace=True)
            )  # 128*32x32
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True),
            nn.LeakyReLU(0.2, inplace=True)
            )  # 256*16x16
        self.conv5 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True),
            nn.LeakyReLU(0.2, inplace=True)
            )  # 256*16x16
   
        
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
        out = self.conv5(out)
        h = self.fl(out) # flatten as 1024*4*4
        z, mu, logvar = self.bottleneck(h)
        return z, mu, logvar


class CVAE_Decoder(nn.Module):
    def __init__(self, h_dim=4096*2*2, z_dim=opt.latent_dim):
        super(CVAE_Decoder, self).__init__()
        self.fc3 = nn.Linear(z_dim, h_dim) # 1024*4*4
        self.unfl = UnFlatten()
        self.conv1 = nn.Sequential(nn.ConvTranspose2d(4096,2048,kernel_size=4,stride=2,padding=1),
            nn.BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True),
            nn.LeakyReLU(0.2, inplace=True)
        )  # 512*8x8        
        self.conv2 = nn.Sequential(nn.ConvTranspose2d(2048,1024,kernel_size=4,stride=2,padding=1),
            nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True),
            nn.LeakyReLU(0.2, inplace=True)
        )  # 512*8x8
        self.conv3 = nn.Sequential(nn.ConvTranspose2d(1024,512,kernel_size=4,stride=2,padding=1),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True),
            nn.LeakyReLU(0.2, inplace=True)
        )  # 256*16x16
        self.conv4 = nn.Sequential(nn.ConvTranspose2d(512,256,kernel_size=4,stride=2,padding=1),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True),
            nn.LeakyReLU(0.2, inplace=True)
        )  # 128*32x32
        self.conv5 = nn.Sequential(nn.ConvTranspose2d(256,128,kernel_size=4,stride=2,padding=1),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True),
            nn.LeakyReLU(0.2, inplace=True)
        )  # 64*64x64
        self.conv6 = nn.Sequential(nn.ConvTranspose2d(128,64,kernel_size=4,stride=2,padding=1),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True),
            nn.LeakyReLU(0.2, inplace=True)
        )  # 32*128x128
        self.conv7 = nn.Sequential(nn.ConvTranspose2d(64,3,kernel_size=4,stride=2,padding=1),
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
        out = self.conv6(out) # 3*256*256
        out = self.conv7(out) # 3*256*256
        return out


class ConvVAE(nn.Module):
    def __init__(self):
        super(ConvVAE, self).__init__()
        self.encoder = CVAE_Encoder()
        
        self.decoder = CVAE_Decoder()

    def forward(self, x):
        z, mu, logvar = self.encoder(x)
        return self.decoder(z), mu, logvar 


