# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Loss functions used in the paper
"Elucidating the Design Space of Diffusion-Based Generative Models"."""

import torch
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from pytorch_msssim import ssim, ms_ssim
from Config import dic_obj as opt

#----------------------------------------------------------------------------
# Loss function corresponding to the variance preserving (VP) formulation
# from the paper "Score-Based Generative Modeling through Stochastic
# Differential Equations".

class VPLoss:
    def __init__(self, beta_d=19.9, beta_min=0.1, epsilon_t=1e-5):
        self.beta_d = beta_d
        self.beta_min = beta_min
        self.epsilon_t = epsilon_t

    def __call__(self, net, images, labels, augment_pipe=None):
        rnd_uniform = torch.rand([images.shape[0], 1, 1, 1], device=images.device)
        sigma = self.sigma(1 + rnd_uniform * (self.epsilon_t - 1))
        weight = 1 / sigma ** 2
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        n = torch.randn_like(y) * sigma
        D_yn = net(y + n, sigma, labels, augment_labels=augment_labels)
        loss = weight * ((D_yn - y) ** 2)
        return loss

    def sigma(self, t):
        t = torch.as_tensor(t)
        return ((0.5 * self.beta_d * (t ** 2) + self.beta_min * t).exp() - 1).sqrt()

#----------------------------------------------------------------------------
# Loss function corresponding to the variance exploding (VE) formulation
# from the paper "Score-Based Generative Modeling through Stochastic
# Differential Equations".

class VELoss:
    def __init__(self, sigma_min=0.002, sigma_max=2):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def __call__(self, net, images, labels, augment_pipe=None):
        rnd_uniform = torch.rand([images.shape[0], 1, 1, 1], device=images.device)
        sigma = self.sigma_min * ((self.sigma_max / self.sigma_min) ** rnd_uniform)
        weight = 1 / sigma ** 2
        #weight = 1
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        n = torch.randn_like(y) * sigma
        D_yn = net(y + n, sigma, labels, augment_labels=augment_labels)
        loss = weight * ((D_yn - y) ** 2)
        return loss

#----------------------------------------------------------------------------
# Improved loss function proposed in the paper "Elucidating the Design Space
# of Diffusion-Based Generative Models" (EDM).

class EDMLoss:
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=0.5):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data

    def __call__(self, net, images, labels=None, augment_pipe=None):
        rnd_normal = torch.randn([images.shape[0], 1, 1, 1], device=images.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        n = torch.randn_like(y) * sigma
        D_yn = net(y + n, sigma, labels, augment_labels=augment_labels)
        loss = weight * ((D_yn - y) ** 2)
        return loss


#----------------------------------------------------------------------------
# Consistency Training Loss (CT) function proposed in the paper "Consitency Models".

class CTLoss:
    def __init__(self, cfg, sigma_data=0.5):
        self.sigma_data = sigma_data
        if cfg.diffusion.ct_dist_fn == 'l2':
            self.loss_fn = torch.nn.functional.mse_loss
        elif cfg.diffusion.ct_dist_fn == 'lpips':
            self.loss_fn = LearnedPerceptualImagePatchSimilarity(net_type='vgg')
        else:
            raise ValueError(f'Loss Function not supported')
        

    def __call__(self, net, net_ema, images, k, labels=None, augment_pipe=None):
        n = torch.randint(1, net.N(k), (images.shape[0],))
        
        weight = 1 # lambda(t) in the paper
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)

        z = torch.randn_like(y) 

        tn_1 = net.t(n + 1, k).reshape(-1, 1, 1, 1).to(images.device)
        f_theta = net(y + tn_1 * z, tn_1, labels, augment_labels=augment_labels)

        with torch.no_grad():
            tn = net.t(n, k).reshape(-1, 1, 1, 1).to(images.device)
            f_theta_ema = net_ema(y + tn * z, tn, labels, augment_labels=augment_labels)

        if isinstance(self.loss_fn, LearnedPerceptualImagePatchSimilarity) and self.loss_fn.device != f_theta.device:
            self.loss_fn.to(f_theta.device)
        loss = weight * self.loss_fn(f_theta.clip(-1, 1), f_theta_ema.clip(-1, 1))
        return loss

#----------------------------------------------------------------------------

# Consistency Distillation Training Loss (CD) function proposed in the paper "Consitency Distillation Models".

class CDLoss:
    def __init__(self, cfg, sigma_data=0.5):
        self.sigma_data = sigma_data
        if cfg.diffusion.ct_dist_fn == 'l2':
            self.loss_fn = torch.nn.functional.mse_loss
        elif cfg.diffusion.ct_dist_fn == 'lpips':
            self.loss_fn = LearnedPerceptualImagePatchSimilarity(net_type='vgg')
        elif cfg.diffusion.ct_dist_fn == 'ms_ssim':
            self.loss_fn = ssim   
        else:
            raise ValueError(f'Loss Function not supported')
        

    def __call__(self, net, net_ema, decoder, images, k, labels=None, augment_pipe=None):
        n = torch.randint(1, net.N(k), (images.shape[0],))

        
        weight = 1 # lambda(t) in the paper
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)


        z = torch.randn_like(y) 

        tn_1 = net.t(n + 1, k).reshape(-1, 1, 1, 1).to(images.device)

        f_theta = net(y + tn_1 * z, tn_1, labels, augment_labels=augment_labels)

        with torch.no_grad():
            tn = net.t(n, k).reshape(-1, 1, 1, 1).to(images.device)
            f_theta_ema = net_ema(y + tn * z, tn, labels, augment_labels=augment_labels)

        if isinstance(self.loss_fn, LearnedPerceptualImagePatchSimilarity) and self.loss_fn.device != f_theta.device:
            self.loss_fn.to(f_theta.device)
        loss =  weight * self.loss_fn(f_theta.clip(-1, 1), f_theta_ema.clip(-1, 1))
        return loss

#----------------------------------------------------------------------------
