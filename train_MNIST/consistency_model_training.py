import json
import os
from dataclasses import asdict, dataclass
from typing import Any, Callable, List, Optional, Tuple, Union

import torch
import numpy as np
import struct
from einops import rearrange
from einops.layers.torch import Rearrange
from lightning import LightningDataModule, LightningModule, Trainer, seed_everything
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
from matplotlib import pyplot as plt
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchinfo import summary
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid
from Config import dic_obj as optc

from consistency_models import (
    ConsistencySamplingAndEditing,
    ConsistencyTraining,
    ema_decay_rate_schedule,
)
from utils import update_ema_model_
from model import Generator_DC,  CVAE_Encoder


def float_to_bits(z):
    if z.dtype != torch.float32:
        raise ValueError("Input tensor must be of type float32.")

    shape = z.shape
    z_flat = z.view(-1)

    bits_tensor = torch.empty(z_flat.size(0), 32, dtype=torch.int32)

    for i, float_number in enumerate(z_flat):

        packed = struct.pack('!f', float_number)  
        for j in range(32):
            bits_tensor[i, j] = (packed[j // 8] >> (7 - (j % 8))) & 1

    return bits_tensor.view(*shape, 32)

def bits_to_float(bits_tensor):
    if bits_tensor.dtype != torch.int32 or bits_tensor.size(-1) != 32:
        raise ValueError("Input tensor must be of type int32 and last dimension must be 32.")

    shape = bits_tensor.shape[:-1]
    float_tensor = torch.empty(shape, dtype=torch.float32)

    for i in range(bits_tensor.size(0)):
        for j in range(bits_tensor.size(1)):
            for p in range(bits_tensor.size(2)):
                for q in range(bits_tensor.size(3)):
                
                    int_value = 0
                    for k in range(32):
                        int_value |= (bits_tensor[i, j, p, q, k].item() << (31 - k))
                    float_tensor[i, j, p, q] = struct.unpack('!f', struct.pack('!I', int_value))[0]
    return float_tensor




def modulate(bits_tensor, modulation_order):

    if modulation_order not in [16, 64, 256]:
        raise ValueError("modulation_order must be one of [16, 64, 256].")
    
    bits_per_symbol = modulation_order.bit_length() - 1  # 16: 4 bits, 64: 6 bits, 256: 8 bits

    shape = bits_tensor.shape 
    num_symbols = shape[0] * shape[1] * shape[2] * shape[3] * shape[4]// bits_per_symbol

    symbols = torch.empty(num_symbols, dtype=torch.complex64)

    for i in range(num_symbols):

        bits_slice = bits_tensor.view(-1)[i * bits_per_symbol:(i + 1) * bits_per_symbol]

        symbol_index = int(''.join(map(str, bits_slice.int().tolist())), 2)

        if modulation_order == 16:
            # 16-QAM
            real_part = (symbol_index % 4) * 2 - 3  # [-3, -1, 1, 3]
            imag_part = (symbol_index // 4) * 2 - 3  # [-3, -1, 1, 3]
        elif modulation_order == 64:
            # 64-QAM
            real_part = (symbol_index % 8) * 2 - 7  # [-7, -5, -3, -1, 1, 3, 5, 7]
            imag_part = (symbol_index // 8) * 2 - 7  # [-7, -5, -3, -1, 1, 3, 5, 7]
        elif modulation_order == 256:
            # 256-QAM
            real_part = (symbol_index % 16) * 2 - 15  # [-15, -13, -11, -9, -7, -5, -3, -1, 1, 3, 5, 7, 9, 11, 13, 15]
            imag_part = (symbol_index // 16) * 2 - 15 # [-15, -13, -11, -9, -7, -5, -3, -1, 1, 3, 5, 7, 9, 11, 13, 15]
        
        symbols[i] = complex(real_part, imag_part)
        

    return symbols.view(*shape[:-1], 32//bits_per_symbol)

def demodulate(symbols, modulation_order):

    if modulation_order not in [16, 64, 256]:
        raise ValueError("modulation_order must be one of [16, 64, 256].")
    
    shape = symbols.shape 
    bits_per_symbol = int(modulation_order.bit_length() - 1)  # 16: 4 bits, 64: 6 bits, 256: 8 bits
    num_symbols = shape[0]*shape[1]*shape[2]*shape[3]*shape[4]

 
    bits_tensor = torch.empty((shape[0]*shape[1]*shape[2]*shape[3]*shape[4]*bits_per_symbol), dtype=torch.int32)


    for i in range(num_symbols):
        symbol = symbols.view(-1)[i]
        if modulation_order == 16:
            # 16-QAM
            real_part = symbol.real
            imag_part = symbol.imag
            symbol_index = (((real_part + 3) / 2).round() % 4) + (((imag_part + 3) / 2).round() % 4) * 4
        elif modulation_order == 64:
            # 64-QAM
            real_part = symbol.real
            imag_part = symbol.imag
            symbol_index = (((real_part + 7) / 2).round() % 8) + (((imag_part + 7) / 2).round() % 8) * 8
        elif modulation_order == 256:
            # 256-QAM
            real_part = symbol.real
            imag_part = symbol.imag
            symbol_index = (((real_part + 15) / 2).round() % 16) + (((imag_part + 15) / 2).round() % 16) * 16
        

        bits_string = format(int(symbol_index), f'0{bits_per_symbol}b')
        for j in range(bits_per_symbol):
            bits_tensor[ i * bits_per_symbol + j] = int(bits_string[j])

    return bits_tensor.view(shape[0], shape[1], shape[2], shape[3], shape[4]*bits_per_symbol)






def awgn_channel(signal, snr_db):
    shape = signal.shape
    x = signal.view(-1)
    snr_linear = 10 ** (snr_db / 10)  
    power_signal = torch.mean(torch.abs(x) ** 2)  
    noise_var = power_signal / snr_linear  
    noise = torch.sqrt(noise_var) * (torch.randn_like(x) + 1j * torch.randn_like(x)) /np.sqrt(2)
    length = x.size(0)
    H_z = torch.eye(2*length)
    H_n = torch.eye(2*length)

    
    y_eq = x + noise
    received_signal = y_eq 

    return received_signal.view(shape), H_z, H_n, noise_var



def rayleigh_channel(signal, snr_db):
    shape = signal.shape
    x = signal.view(-1)
    h = (torch.randn(x.size()) + 1j * torch.randn(x.size())) / np.sqrt(2)
    snr_linear = 10 ** (snr_db / 10)  
    power_signal = torch.mean(torch.abs(x) ** 2)  
    noise_var = power_signal / snr_linear  
    noise = torch.sqrt(noise_var) * (torch.randn_like(x) + 1j * torch.randn_like(x)) /np.sqrt(2)
    
    length = x.size(0)
    H_z = torch.eye(2*length)
    H_n = torch.eye(2*length)
    for i in range(length):
        H_z[i,i] = ((torch.abs(h[i]))**2) / ((torch.abs(h[i]))**2 + noise_var )
        H_z[i+length,i+length] = H_z[i,i]
        H_n[i,i] = h[i].real / ((torch.abs(h[i]))**2 + noise_var )
        H_n[i+length,i+length] = (h[i].imag) / ((torch.abs(h[i]))**2 + noise_var )
    
    y_eq = torch.zeros_like(x)
    for i in range(length):
        y_eq[i] = (x[i].real)* H_z[i,i] + 1j*((x[i].imag)*H_z[i+length,i+length]) + (noise[i].real)* H_n[i,i] + 1j*((noise[i].imag)*H_n[i+length,i+length])
    received_signal = y_eq 
    
    return received_signal.view(shape), H_z, H_n, noise_var




def rician_channel(signal, k_factor, snr_db):
    k_factor = torch.tensor(k_factor)
    shape = signal.shape
    x = signal.view(-1)    
    direct_component = torch.sqrt(k_factor / (k_factor + 1)) * torch.ones_like(x)
    scattered_component =torch.sqrt(1 / (k_factor + 1)) * (torch.randn(x.size()) + 1j * torch.randn(x.size())) / np.sqrt(2)
    h = direct_component + scattered_component
    snr_linear = 10 ** (snr_db / 10)  
    power_signal = torch.mean(torch.abs(x) ** 2)  
    noise_var = power_signal / snr_linear  
    noise = torch.sqrt(noise_var) * (torch.randn_like(x) + 1j * torch.randn_like(x)) /np.sqrt(2)    
    
    length = x.size(0)
    H_z = torch.eye(2*length)
    H_n = torch.eye(2*length)
    for i in range(length):
        H_z[i,i] = ((torch.abs(h[i]))**2) / ((torch.abs(h[i]))**2 + noise_var )
        H_z[i+length,i+length] = H_z[i,i]
        H_n[i,i] = h[i].real / ((torch.abs(h[i]))**2 + noise_var )
        H_n[i+length,i+length] = (h[i].imag) / ((torch.abs(h[i]))**2 + noise_var )
    
    y_eq = torch.zeros_like(x)
    for i in range(length):
        y_eq[i] = (x[i].real)* H_z[i,i] + 1j*((x[i].imag)*H_z[i+length,i+length]) + (noise[i].real)* H_n[i,i] + 1j*((noise[i].imag)*H_n[i+length,i+length])
    received_signal = y_eq 
    
    return received_signal.view(shape), H_z, H_n, noise_var




def simulate_wireless_channel(z, M, channel_type, snr_db=None, K=None):

    bits = float_to_bits(z)

    symbols = modulate(bits, M)

    if channel_type == 'AWGN':
        received_symbols, H_z, H_n, noise_var = awgn_channel(signal = symbols, snr_db = snr_db)
    elif channel_type == 'Rayleigh':
        received_symbols, H_z, H_n, noise_var = rayleigh_channel(signal = symbols, snr_db = snr_db)
    elif channel_type == 'Rician':
        received_symbols, H_z, H_n, noise_var = rician_channel(signal = symbols, k_factor = 2, snr_db = snr_db)
    else:
        raise ValueError("Unsupported channel type")

    return received_symbols, H_z, H_n


def simulate_demodulate(received_symbols, M):
    recons_bits_tensor = demodulate(received_symbols, M)
    recons_z = bits_to_float(recons_bits_tensor) 
    
    
    return recons_z   





@dataclass
class ImageDataModuleConfig:
    data_dir: str = "butterflies256"
    image_size: Tuple[int, int] = (optc.latent_size_H, optc.latent_size_W)
    batch_size: int = 32
    num_workers: int = optc.latent_size_C
    pin_memory: bool = True
    persistent_workers: bool = True


class ImageDataModule(LightningDataModule):
    def __init__(self, config: ImageDataModuleConfig) -> None:
        super().__init__()

        self.config = config

    def setup(self, stage: str = None) -> None:
        transform = T.Compose(
            [
                T.Resize(self.config.image_size),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Lambda(lambda x: (x * 2) - 1),
            ]
        )
        self.dataset = ImageFolder(self.config.data_dir, transform=transform)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            persistent_workers=self.config.persistent_workers,
        )
    

def GroupNorm(channels: int) -> nn.GroupNorm:
    return nn.GroupNorm(num_groups=min(32, channels // 4), num_channels=channels)


class SelfAttention(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_heads: int = 8,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        self.dropout = dropout

        self.qkv_projection = nn.Sequential(
            GroupNorm(in_channels),
            nn.Conv2d(in_channels, 3 * in_channels, kernel_size=1, bias=False),
            Rearrange("b (i h d) x y -> i b h (x y) d", i=3, h=n_heads),
        )
        self.output_projection = nn.Sequential(
            Rearrange("b h l d -> b l (h d)"),
            nn.Linear(in_channels, out_channels, bias=False),
            Rearrange("b l d -> b d l"),
            GroupNorm(out_channels),
            nn.Dropout1d(dropout),
        )
        self.residual_projection = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        q, k, v = self.qkv_projection(x).unbind(dim=0)

        output = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if self.training else 0.0, is_causal=False
        )
        output = self.output_projection(output)
        output = rearrange(output, "b c (x y) -> b c x y", x=x.shape[-2], y=x.shape[-1])

        return output + self.residual_projection(x)


class UNetBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        noise_level_channels: int,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        self.input_projection = nn.Sequential(
            GroupNorm(in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding="same"),
            nn.Dropout2d(dropout),
        )
        self.noise_level_projection = nn.Sequential(
            nn.SiLU(),
            nn.Conv2d(noise_level_channels, out_channels, kernel_size=1),
        )
        self.output_projection = nn.Sequential(
            GroupNorm(out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding="same"),
            nn.Dropout2d(dropout),
        )
        self.residual_projection = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: Tensor, noise_level: Tensor) -> Tensor:
        h = self.input_projection(x)
        h = h + self.noise_level_projection(noise_level)

        return self.output_projection(h) + self.residual_projection(x)


class UNetBlockWithSelfAttention(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        noise_level_channels: int,
        n_heads: int = 8,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        self.unet_block = UNetBlock(
            in_channels, out_channels, noise_level_channels, dropout
        )
        self.self_attention = SelfAttention(
            out_channels, out_channels, n_heads, dropout
        )

    def forward(self, x: Tensor, noise_level: Tensor) -> Tensor:
        return self.self_attention(self.unet_block(x, noise_level))


class Downsample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()

        self.projection = nn.Sequential(
            Rearrange("b c (h ph) (w pw) -> b (c ph pw) h w", ph=2, pw=2),
            nn.Conv2d(4 * channels, channels, kernel_size=1),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.projection(x)


class Upsample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()

        self.projection = nn.Sequential(
            nn.Upsample(scale_factor=2.0, mode="nearest"),
            nn.Conv2d(channels, channels, kernel_size=3, padding="same"),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.projection(x)


class NoiseLevelEmbedding(nn.Module):
    def __init__(self, channels: int, scale: float = 16.0) -> None:
        super().__init__()

        self.W = nn.Parameter(torch.randn(channels // 2) * scale, requires_grad=False)

        self.projection = nn.Sequential(
            nn.Linear(channels, 4 * channels),
            nn.SiLU(),
            nn.Linear(4 * channels, channels),
            Rearrange("b c -> b c () ()"),
        )

    def forward(self, x: Tensor) -> Tensor:
        h = x[:, None] * self.W[None, :] * 2 * torch.pi
        h = torch.cat([torch.sin(h), torch.cos(h)], dim=-1)

        return self.projection(h)
    


@dataclass
class UNetConfig:
    channels: int = optc.latent_size_C
    noise_level_channels: int = 256
    noise_level_scale: float = 16.0
    n_heads: int = 8
    top_blocks_channels: Tuple[int, ...] = (128, 128)
    top_blocks_n_blocks_per_resolution: Tuple[int, ...] = (2, 2)
    top_blocks_has_resampling: Tuple[bool, ...] = (True, True)
    top_blocks_dropout: Tuple[float, ...] = (0.0, 0.0)
    mid_blocks_channels: Tuple[int, ...] = (256, 512)
    mid_blocks_n_blocks_per_resolution: Tuple[int, ...] = (4, 4)
    mid_blocks_has_resampling: Tuple[bool, ...] = (True, False)
    mid_blocks_dropout: Tuple[float, ...] = (0.0, 0.0)


class UNet(nn.Module):
    def __init__(self, config: UNetConfig) -> None:
        super().__init__()

        self.config = config

        self.input_projection = nn.Conv2d(
            config.channels,
            config.top_blocks_channels[0],
            kernel_size=3,
            padding="same",
        )
        self.noise_level_embedding = NoiseLevelEmbedding(
            config.noise_level_channels, config.noise_level_scale
        )
        self.top_encoder_blocks = self._make_encoder_blocks(
            self.config.top_blocks_channels + self.config.mid_blocks_channels[:1],
            self.config.top_blocks_n_blocks_per_resolution,
            self.config.top_blocks_has_resampling,
            self.config.top_blocks_dropout,
            self._make_top_block,
        )
        self.mid_encoder_blocks = self._make_encoder_blocks(
            self.config.mid_blocks_channels + self.config.mid_blocks_channels[-1:],
            self.config.mid_blocks_n_blocks_per_resolution,
            self.config.mid_blocks_has_resampling,
            self.config.mid_blocks_dropout,
            self._make_mid_block,
        )
        self.mid_decoder_blocks = self._make_decoder_blocks(
            self.config.mid_blocks_channels + self.config.mid_blocks_channels[-1:],
            self.config.mid_blocks_n_blocks_per_resolution,
            self.config.mid_blocks_has_resampling,
            self.config.mid_blocks_dropout,
            self._make_mid_block,
        )
        self.top_decoder_blocks = self._make_decoder_blocks(
            self.config.top_blocks_channels + self.config.mid_blocks_channels[:1],
            self.config.top_blocks_n_blocks_per_resolution,
            self.config.top_blocks_has_resampling,
            self.config.top_blocks_dropout,
            self._make_top_block,
        )
        self.output_projection = nn.Conv2d(
            config.top_blocks_channels[0],
            config.channels,
            kernel_size=3,
            padding="same",
        )

    def forward(self, x: Tensor, noise_level: Tensor) -> Tensor:
        h = self.input_projection(x)
        noise_level = self.noise_level_embedding(noise_level)

        top_encoder_embeddings = []
        for block in self.top_encoder_blocks:
            if isinstance(block, UNetBlock):
                h = block(h, noise_level)
                top_encoder_embeddings.append(h)
            else:
                h = block(h)

        mid_encoder_embeddings = []
        for block in self.mid_encoder_blocks:
            if isinstance(block, UNetBlockWithSelfAttention):
                h = block(h, noise_level)
                mid_encoder_embeddings.append(h)
            else:
                h = block(h)

        for block in self.mid_decoder_blocks:
            if isinstance(block, UNetBlockWithSelfAttention):
                h = torch.cat((h, mid_encoder_embeddings.pop()), dim=1)
                h = block(h, noise_level)
            else:
                h = block(h)

        for block in self.top_decoder_blocks:
            if isinstance(block, UNetBlock):
                h = torch.cat((h, top_encoder_embeddings.pop()), dim=1)
                h = block(h, noise_level)
            else:
                h = block(h)

        return self.output_projection(h)

    def _make_encoder_blocks(
        self,
        channels: Tuple[int, ...],
        n_blocks_per_resolution: Tuple[int, ...],
        has_resampling: Tuple[bool, ...],
        dropout: Tuple[float, ...],
        block_fn: Callable[[], nn.Module],
    ) -> nn.ModuleList:
        blocks = nn.ModuleList()

        channel_pairs = list(zip(channels[:-1], channels[1:]))
        for idx, (in_channels, out_channels) in enumerate(channel_pairs):
            for _ in range(n_blocks_per_resolution[idx]):
                blocks.append(block_fn(in_channels, out_channels, dropout[idx]))
                in_channels = out_channels

            if has_resampling[idx]:
                blocks.append(Downsample(out_channels))

        return blocks

    def _make_decoder_blocks(
        self,
        channels: Tuple[int, ...],
        n_blocks_per_resolution: Tuple[int, ...],
        has_resampling: Tuple[bool, ...],
        dropout: Tuple[float, ...],
        block_fn: Callable[[], nn.Module],
    ) -> nn.ModuleList:
        blocks = nn.ModuleList()

        channel_pairs = list(zip(channels[:-1], channels[1:]))[::-1]
        for idx, (out_channels, in_channels) in enumerate(channel_pairs):
            if has_resampling[::-1][idx]:
                blocks.append(Upsample(in_channels))

            inner_blocks = []
            for _ in range(n_blocks_per_resolution[::-1][idx]):
                inner_blocks.append(
                    block_fn(in_channels * 2, out_channels, dropout[::-1][idx])
                )
                out_channels = in_channels
            blocks.extend(inner_blocks[::-1])

        return blocks

    def _make_top_block(
        self, in_channels: int, out_channels: int, dropout: float
    ) -> UNetBlock:
        return UNetBlock(
            in_channels,
            out_channels,
            self.config.noise_level_channels,
            dropout,
        )

    def _make_mid_block(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float,
    ) -> UNetBlockWithSelfAttention:
        return UNetBlockWithSelfAttention(
            in_channels,
            out_channels,
            self.config.noise_level_channels,
            self.config.n_heads,
            dropout,
        )

    def save_pretrained(self, pretrained_path: str) -> None:
        os.makedirs(pretrained_path, exist_ok=True)

        with open(os.path.join(pretrained_path, "config.json"), mode="w") as f:
            json.dump(asdict(self.config), f)

        torch.save(self.state_dict(), os.path.join(pretrained_path, "model.pt"))

    @classmethod
    def from_pretrained(cls, pretrained_path: str) -> "UNet":
        with open(os.path.join(pretrained_path, "config.json"), mode="r") as f:
            config_dict = json.load(f)
        config = UNetConfig(**config_dict)

        model = cls(config)

        state_dict = torch.load(
            os.path.join(pretrained_path, "model.pt"), map_location=torch.device("cpu")
        )
        model.load_state_dict(state_dict)

        return model

from diffusers import AutoencoderKL 

def normalize_fun(x):

    x_normalized = torch.zeros_like(x)
    

    max_vals = []
    min_vals = []
    

    for c in range(x.size(1)): 

        x_channel = x[:, c, :, :]


        min_val = torch.min(x_channel)
        max_val = torch.max(x_channel)
        min_vals.append(min_val.item())
        max_vals.append(max_val.item())

        x_channel_normalized = (x_channel - min_val) / (max_val - min_val)

        x_channel_normalized = (x_channel_normalized * 2) - 1

        x_normalized[:, c, :, :] = x_channel_normalized

 
    return x_normalized, max_vals, min_vals

def denormalize_tensor(x_normalized, max_vals, min_vals):
    x_denormalized = torch.zeros_like(x_normalized)
    

    for c in range(x_normalized.size(1)):  
        x_channel_normalized = x_normalized[:, c, :, :]
        
        x_channel = (x_channel_normalized + 1) / 2 * (max_vals[c] - min_vals[c]) + min_vals[c]
        
        x_denormalized[:, c, :, :] = x_channel
        
    return x_denormalized


@dataclass
class LitConsistencyModelConfig:
    initial_ema_decay_rate: float = 0.95
    student_model_ema_decay_rate: float = 0.99993
    lr: float = 1e-4
    betas: Tuple[float, float] = (0.9, 0.995)
    lr_scheduler_start_factor: float = 1e-5
    lr_scheduler_iters: int = 10_000
    sample_every_n_steps: int = 10_000
    num_samples: int = 8
    sampling_sigmas: Tuple[Tuple[int, ...], ...] = (
        (80,),
        (80.0, 0.661),
        (80.0, 24.4, 5.84, 0.9, 0.661),
    )


class LitConsistencyModel(LightningModule):
    def __init__(
        self,
        consistency_training: ConsistencyTraining,
        consistency_sampling: ConsistencySamplingAndEditing,
        student_model: UNet,
        teacher_model: UNet,
        ema_student_model: UNet,
        config: LitConsistencyModelConfig,
    ) -> None:
        super().__init__()

        self.consistency_training = consistency_training
        self.consistency_sampling = consistency_sampling
        self.student_model = student_model
        self.teacher_model = teacher_model
        self.ema_student_model = ema_student_model
        self.config = config
        self.num_timesteps = self.consistency_training.initial_timesteps

        #Load the model

        self.lpips = LearnedPerceptualImagePatchSimilarity(net_type="alex")
        for param in self.lpips.parameters():
             param.requires_grad = False 
        self.lpips.eval() 

    
        self.E = CVAE_Encoder().cuda()
        self.E.load_state_dict(torch.load(f'd:\\Python\\SemCom\\saved_model\\v_encoder_dc_model_of_MNIST.pth'))
        for param in self.E.parameters():
            param.requires_grad = False  
        self.E.eval()
        
        self.G = Generator_DC(input_dim = optc.latent_dim).cuda()
        self.G.load_state_dict(torch.load(f'd:\\Python\\SemCom\\saved_model\\generator_dc_model_of_MNIST.pth'))
        for param in self.E.parameters():
            param.requires_grad = False  
        self.G.eval()        
 

        # Freeze teacher and EMA student models and set to eval mode
        for param in self.teacher_model.parameters():
            param.requires_grad = False
        for param in self.ema_student_model.parameters():
            param.requires_grad = False
        self.teacher_model = self.teacher_model.eval()
        self.ema_student_model = self.ema_student_model.eval()

    def training_step(self, batch: Union[Tensor, List[Tensor]], batch_idx: int) -> None:
        if isinstance(batch, list):
            batch = batch[0]

        batch, mean_val, std_val = self.E(batch)
        batch = batch.reshape(batch.shape[0], optc.latent_size_C, optc.latent_size_H, optc.latent_size_W)

        batch, max_vals, min_vals = normalize_fun(batch)

        #wireless channel part
        modulation_order = 256
        batch, H_z, H_n = simulate_wireless_channel(batch, M = modulation_order, channel_type = 'AWGN', snr_db = 100, K=2)
        bits_per_symbol = int(modulation_order.bit_length() - 1) 
        batch = batch.reshape(batch.shape[0], optc.latent_size_C*32//bits_per_symbol, optc.latent_size_H, optc.latent_size_W)

        output = self.consistency_training(
            self.student_model,
            self.teacher_model,
            batch,
            self.global_step,
            self.trainer.max_steps,
        )
        self.num_timesteps = output.num_timesteps

        output1 = simulate_demodulate(received_symbols = output.predicted, M = modulation_order)
        output2 = simulate_demodulate(received_symbols = output.target, M = modulation_order)

        output1 = denormalize_tensor((output1), max_vals, min_vals)
        output2 = denormalize_tensor((output1), max_vals, min_vals)

        lpips_loss =  self.lpips(
           self.G(output1.reshape(batch.shape[0], optc.latent_dim)) , self.G(output2.reshape(batch.shape[0], optc.latent_dim))
        )
        overflow_loss = self.lpips(
             self.G(output1.reshape(batch.shape[0], optc.latent_dim)), self.G(output2.reshape(batch.shape[0], optc.latent_dim)).detach().clamp(-1.0, 1.0)
        )
        loss = lpips_loss + overflow_loss
        print(loss)

        self.log_dict(
            {
                "train_loss": loss,
                "lpips_loss": lpips_loss,
                "overflow_loss": overflow_loss,
                "num_timesteps": output.num_timesteps,
            }
        )

        return loss

    def on_train_batch_end(
        self, outputs: Any, batch: Union[Tensor, List[Tensor]], batch_idx: int
    ) -> None:
        # Update teacher model
        ema_decay_rate = ema_decay_rate_schedule(
            self.num_timesteps,
            self.config.initial_ema_decay_rate,
            self.consistency_training.initial_timesteps,
        )
        update_ema_model_(self.teacher_model, self.student_model, ema_decay_rate)
        self.log_dict({"ema_decay_rate": ema_decay_rate})

        # Update EMA student model
        update_ema_model_(
            self.ema_student_model,
            self.student_model,
            self.config.student_model_ema_decay_rate,
        )

        if (
            (self.global_step + 1) % self.config.sample_every_n_steps == 0
        ) or self.global_step == 0:
            self.__sample_and_log_samples(batch)

    def configure_optimizers(self):
        opt = torch.optim.Adam(
            self.student_model.parameters(), lr=self.config.lr, betas=self.config.betas
        )
        sched = torch.optim.lr_scheduler.LinearLR(
            opt,
            start_factor=self.config.lr_scheduler_start_factor,
            total_iters=self.config.lr_scheduler_iters,
        )
        sched = {"scheduler": sched, "interval": "step", "frequency": 1}

        return [opt], [sched]

    @torch.no_grad()
    def __sample_and_log_samples(self, batch: Union[Tensor, List[Tensor]]) -> None:
        if isinstance(batch, list):
            batch = batch[0]

        # Ensure the number of samples does not exceed the batch size
        num_samples = min(self.config.num_samples, batch.shape[0])
        noise = torch.randn_like(batch[:num_samples])

        # Log ground truth samples
        self.__log_images(
            batch[:num_samples].detach().clone(), f"ground_truth", self.global_step
        )

        for sigmas in self.config.sampling_sigmas:
            samples = self.consistency_sampling(
                self.ema_student_model, noise, sigmas, clip_denoised=True, verbose=True
            )
            samples = samples.clamp(min=-1.0, max=1.0)

            # Generated samples
            self.__log_images(
                samples,
                f"generated_samples-sigmas={sigmas}",
                self.global_step,
            )

    @torch.no_grad()
    def __log_images(self, images: Tensor, title: str, global_step: int) -> None:
        images = images.detach().float()

        grid = make_grid(
            images.clamp(-1.0, 1.0), value_range=(-1.0, 1.0), normalize=True
        )
        self.logger.experiment.add_image(title, grid, global_step)




import glob
from PIL import Image
from torchvision import datasets, transforms
from torch.utils.data import Dataset
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

@dataclass
class TrainingConfig:
    image_dm_config: ImageDataModuleConfig
    unet_config: UNetConfig
    consistency_training: ConsistencyTraining
    consistency_sampling: ConsistencySamplingAndEditing
    lit_cm_config: LitConsistencyModelConfig
    trainer: Trainer
    seed: int = 66
    model_ckpt_path: str = "D:\\Python\\SemCom\\saved_model\\new_MNIST"
    resume_ckpt_path: Optional[str] = None




def run_training(config: TrainingConfig) -> None:
    # Set seed
    seed_everything(config.seed)

    # Create data module
    #dm = ImageDataModule(config.image_dm_config)
    #load dataset
    transform = transforms.Compose([transforms.Resize(optc.inputdata_size_H), transforms.ToTensor(), transforms.Normalize(mean=[0.5], std=[0.5])  ])
    #transform = transforms.Compose([transforms.Resize(opt.inputdata_size_H), transforms.CenterCrop(opt.inputdata_size_W), transforms.PILToTensor(), transforms.ConvertImageDtype(torch.float), transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])])

    # use MNIST dataset
    dataset = datasets.MNIST('d:\\Python\\SemCom\\saved_data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size = optc.batch_size_diffusion, shuffle=True)
 

    # Create student and teacher models and EMA student model
    student_model = UNet(config.unet_config)
    teacher_model = UNet(config.unet_config)
    teacher_model.load_state_dict(student_model.state_dict())
    ema_student_model = UNet(config.unet_config)
    ema_student_model.load_state_dict(student_model.state_dict())

    # Create lightning module
    lit_cm = LitConsistencyModel(
        config.consistency_training,
        config.consistency_sampling,
        student_model,
        teacher_model,
        ema_student_model,
        config.lit_cm_config,
    )

    # Run training
    config.trainer.fit(lit_cm, train_loader, ckpt_path=config.resume_ckpt_path)

    # Save model
    lit_cm.ema_student_model.save_pretrained(config.model_ckpt_path)



if __name__ == '__main__':    


    training_config = TrainingConfig(
    image_dm_config=ImageDataModuleConfig("butterflies256"),
    unet_config=UNetConfig(),
    consistency_training=ConsistencyTraining(final_timesteps=100),
    consistency_sampling=ConsistencySamplingAndEditing(),
    lit_cm_config=LitConsistencyModelConfig(
        sample_every_n_steps=20000, lr_scheduler_iters=20000
    ),
    trainer=Trainer(
        max_steps=10000,
        precision="16-mixed",
        log_every_n_steps=10,
        logger=TensorBoardLogger(".", name="logs", version="cm"),
        callbacks=[LearningRateMonitor(logging_interval="step")],
    ),
    )
    run_training(training_config)    