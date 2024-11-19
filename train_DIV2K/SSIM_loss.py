import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp

def type_trans(window,img):
    if img.is_cuda:
        window = window.cuda(img.get_device())
    return window.type_as(img)

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def _ssim(img1, img2, window, window_size, channel, size_average = True):

    mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)
    # print(mu1.shape,mu2.shape)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12   = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
    mcs_map  = (2.0 * sigma12 + C2)/(sigma1_sq + sigma2_sq + C2)
    # print(ssim_map.shape)
    if size_average:
        return ssim_map.mean(), mcs_map.mean()
    # else:
    #     return ssim_map.mean(1).mean(1).mean(1)

class SSIM(torch.nn.Module):
    def __init__(self, window_size = 11, size_average = True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average

    def forward(self, img1, img2):
        _, channel, _, _ = img1.size()
        window = create_window(self.window_size,channel)
        window = type_trans(window,img1)
        ssim_map, mcs_map =_ssim(img1, img2, window, self.window_size, channel, self.size_average)
        return ssim_map


class MS_SSIM(torch.nn.Module):
    def __init__(self, window_size = 11,size_average = True):
        super(MS_SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        # self.channel = 3

    def forward(self, img1, img2, levels=5):

        weight = Variable(torch.Tensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333]))

        msssim = Variable(torch.Tensor(levels,))
        mcs    = Variable(torch.Tensor(levels,))

        if torch.cuda.is_available():
            weight =weight.cuda()
            msssim=msssim.cuda()
            mcs=mcs.cuda()

        _, channel, _, _ = img1.size()
        window = create_window(self.window_size,channel)
        window = type_trans(window,img1)

        for i in range(levels): #5 levels
            ssim_map, mcs_map = _ssim(img1, img2,window,self.window_size, channel, self.size_average)
            msssim[i] = ssim_map
            mcs[i] = mcs_map
            # print(img1.shape)
            filtered_im1 = F.avg_pool2d(img1, kernel_size=2, stride=2)
            filtered_im2 = F.avg_pool2d(img2, kernel_size=2, stride=2)
            img1 = filtered_im1 #refresh img
            img2 = filtered_im2

        return torch.prod((msssim[levels-1]**weight[levels-1] * mcs[0:levels-1]**weight[0:levels-1]))
        # return torch.prod((msssim[levels-1] * mcs[0:levels-1]))
        #torch.prod: Returns the product of all elements in the input tensor
