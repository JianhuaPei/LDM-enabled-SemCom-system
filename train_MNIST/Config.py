
from munch import DefaultMunch


Config={


    'inputdata_size_H': 32,
    'inputdata_size_W': 32, 
    'inputdata_size_C':1,

    'latent_size_H':8,
    'latent_size_W':8,
    'latent_size_C':1,

    'batch_size_GAN':64,
    'epochs_GAN':20000,
    'lr_GAN_D':0.0002, #learning rate of GAN
    'lr_GAN_G':0.0002, #learning rate of GAN
    'latent_dim':8*8, #latent dimension
    'n_critic':5, #the number of training D when training G once
    'clip_value':0.01, # clip D
    'lambda_1':10,

    'balance_alpha1':0.05,
    'balance_alpha2':0.5,
    
    'epochs_VAE':100,
    'batch_size_VAE':128,
    'batch_size_CVAE':128,
    'lr_VAE':0.0001,

    'epochs_adaptor':200,

    
    'weight_decay': 0,
    'lr': 4.e-4,
    'betas': [0.9,0.999],
    'eps': 1.e-8,
    'amsgrad': False,
    'batch_size_diffusion':128,
    'epochs_VE_diffusion':100
    

    }
dic_obj = DefaultMunch.fromDict(Config)
