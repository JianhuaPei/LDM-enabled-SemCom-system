import torch
import numpy as np
from PIL import Image
from io import BytesIO
import imageio




def image_tensor_to_jpeg2000(tensor, compression_ratio):
    pil_img = Image.fromarray(((tensor.cpu().numpy().squeeze()+1) * 127.5).astype(np.uint8))
    buffer = BytesIO()
    pil_img.save(buffer, format='JPEG2000', quality_layers=[compression_ratio])
    return buffer.getvalue()

def jpeg2000_to_image_tensor(jpeg2000_data, channels):
    image = imageio.imread(BytesIO(jpeg2000_data), 'jpeg2000')
    tensor = (torch.from_numpy(image).float() / 127.5) -1
    if channels == 1:
        tensor = tensor.unsqueeze(0)  # Add channel dimension for grayscale
    else:
        tensor = tensor.permute(2, 0, 1)  # Rearrange dimensions for color
    return tensor

from pyldpc import make_ldpc, encode, decode, get_message

def ldpc_encode(data, snr):
    n = len(data) * 8
    d_v, d_c =  4, 8  # LDPC parameters
    H, G = make_ldpc(n, d_v, d_c, systematic=True, sparse=True)
    k = G.shape[1]
    
    # 将数据转换为比特
    bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8))
    # 如果长度不是k的倍数，需要填充
    padding_length = (-len(bits)) % k
    bits = np.concatenate([bits, np.zeros(padding_length, dtype=np.uint8)])
    # 进行LDPC编码
    print(G.shape)
    print(bits.shape)
    encoded_bits = encode(G, bits, snr)
    return encoded_bits.tostring(), padding_length

def ldpc_decode(encoded_data, padding_length, snr):
    n = len(encoded_data) * 8
    d_v, d_c = 4, 8
    H, G = make_ldpc(n, d_v, d_c, systematic=True, sparse=True)
    k = G.shape[1]
    
    # 将数据转换为比特
    encoded_bits = np.frombuffer(encoded_data, dtype=np.uint8)
    encoded_bits = np.unpackbits(encoded_bits)
    # 进行LDPC解码
    decoded_bits, _ = decode(H, encoded_bits, snr, maxiter=100)
    # 移除填充
    decoded_bits = decoded_bits[:len(decoded_bits) - padding_length]
    # 转换回字节
    return np.packbits(decoded_bits).tostring()

def compress_transmit_decompress(tensor, compression_ratio, snr=10):
    batch_size, channels, height, width = tensor.size()
    
    # 压缩图像
    compressed_batch = []
    for i in range(batch_size):
        img_data = image_tensor_to_jpeg2000(tensor[i], compression_ratio)
        encoded_data, padding_length = ldpc_encode(img_data, snr)
        compressed_batch.append((encoded_data, padding_length))
    
    # 模拟传输过程...
    
    # 解压缩图像
    decompressed_batch = []
    for encoded_data, padding_length in compressed_batch:
        decoded_data = ldpc_decode(encoded_data, padding_length, snr)
        img_tensor = jpeg2000_to_image_tensor(decoded_data, channels)
        decompressed_batch.append(img_tensor)
    
    decompressed_tensor = torch.stack(decompressed_batch).to(tensor.device)
    return decompressed_tensor

from glymur import Jp2k
import io
import pyldpc

def jpeg2000_ldpc_compress_decompress(tensor, compression_ratio, snr):
    # 确保输入是PyTorch张量
    tensor = tensor.cpu()
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("Input must be a pytorch tensor")

    # 转换为NumPy数组，JPEG2000库使用NumPy数组
    numpy_tensor = tensor.numpy()
    batch_size, channels, H, W = numpy_tensor.shape

    # 建立一个包含解码图像的空的NumPy数组
    decoded_images = np.empty_like(numpy_tensor)

    for i in range(batch_size):
        # 处理每一张图片
        image = Image.fromarray(np.clip((tensor[i].numpy().squeeze() * 127.5 + 127.5), 0, 255).astype(np.uint8))

        # 使用JPEG2000进行压缩
        with io.BytesIO() as jp2_buffer:
            # 保存JPEG2000压缩后的图像到内存
            image.save(jp2_buffer, format='JPEG2000', quality_layers=[compression_ratio])
            # 读取压缩后的数据
            compressed_data = jp2_buffer.getvalue()

            # LDPC编码
            n = len(compressed_data) * 8
            d_v = 2
            d_c = int(n / (n * (1/compression_ratio) / d_v))
            n=2000
            d_v=3
            d_c=4
            snr_linear = 10 ** (snr / 10)
    
            tG, tH = pyldpc.make_ldpc(n, d_v, d_c, systematic=True, sparse=True)

            k = tG.shape[1]
            v = np.unpackbits(np.frombuffer(compressed_data, dtype=np.uint8))
            padding_length = (-len(v)) % k
            v = np.concatenate([v, np.zeros(padding_length, dtype=np.uint8)])
            y = pyldpc.encode(tG, v, snr=100)
            
            # 模拟信道传输
            sigma = np.sqrt(1 / (2 * snr_linear))
            noise = sigma * np.random.randn(len(y))
            y_noisy = y + noise

            # LDPC解码
            d = pyldpc.decode(tH, y_noisy, maxiter=100, snr=100)
            d = np.packbits(d.astype(np.uint8))

            # 从内存中读取JPEG2000压缩的图像并解码
            with io.BytesIO(d.tobytes()) as jp2_buffer_recovered:
                recovered_image = imageio.imread(BytesIO(jp2_buffer_recovered), 'jpeg2000')


        # 将解压缩的图像放回PyTorch张量中
        decoded_images[i] = torch.from_numpy(np.expand_dims(recovered_image, 0))

    # 将图像重新归一化到[-1, 1]
    decoded_images = (decoded_images / 127.5) - 1.0

    return torch.from_numpy(decoded_images).cuda()

import os
import glymur
from imageio import imwrite
import numpy as np
from pyldpc import make_ldpc, ldpc_images
from pyldpc.utils_img import gray2bin, rgb2bin
from matplotlib import pyplot as plt
from PIL import Image
import math

def save_as_jpeg2000(tensor, compression_ratio, save_dir, filename='compressed.jp2'):
    if not isinstance(tensor, torch.Tensor):
        raise TypeError('Input must be a PyTorch tensor')
    
    if not (tensor.ndim == 3 and tensor.shape[0] in {1, 3}):
        raise ValueError('Input tensor must be of shape (C, H, W) where C is 1 or 3')
    
    if not os.path.isdir(save_dir):
        raise ValueError('Save directory does not exist')

    # 反归一化到0-255
    img = ((tensor.detach().cpu().numpy() + 1) / 2 * 255).astype(np.uint8)

    if img.shape[0] == 1:
        img = np.squeeze(img, axis=0)  # 去除通道维度，因为是灰度图像
    else:
        img = np.transpose(img, (1, 2, 0))  # 转置成(H, W, C)格式
    
    # 设置文件完整路径
    file_path = os.path.join(save_dir, filename)
    
    # 保存图片
    imwrite(file_path, img, format='jp2', compression=compression_ratio)



def JPEG2000_LDPC(tensor, compression_ratio, snr_num):

    tensor = tensor.cpu()
    # 转换为NumPy数组，JPEG2000库使用NumPy数组
    numpy_tensor = tensor.numpy()
    batch_size, channels, H_dim, W_dim = numpy_tensor.shape

    for i in range(batch_size):
        save_dir = 'D:\\Python\\SemCom\\saved_data'
        save_as_jpeg2000(tensor[i], compression_ratio, save_dir)
        filename='compressed.jp2'
        file_path = os.path.join(save_dir, filename)
        if channels == 1:
            image = Image.open(file_path)
            # convert it to grayscale and keep one channel
            image = np.asarray(image.convert('LA'))[:, :, 0]

            # Convert it to a binary matrix
            image_bin = gray2bin(image)
            n = 200
            d_v = 3
            d_c = 4
            snr = 100

            H, G = make_ldpc(n, d_v, d_c, systematic=True, sparse=True)

            image_coded, image_noisy = ldpc_images.encode_img(G, image_bin, snr=100)

            #wireless channel
            #wireless channel
            snr_linear = 10 ** (snr_num / 10)
            var_signal = np.var(image_coded)
            sigma = math.sqrt(var_signal/snr_linear)

            noise = sigma * np.random.randn(*image_coded.shape)
            image_coded_noisy = image_coded + noise


            image_decoded = ldpc_images.decode_img(G, H, image_coded_noisy, snr, image_bin.shape)
            numpy_tensor[i,:,:,:] = image_decoded 


        if channels ==3:
            image = np.asarray(Image.open(file_path))
            # Convert it to a binary matrix
            image_bin = rgb2bin(image)
            n = 200
            d_v = 3
            d_c = 4
            snr = 100

            H, G = make_ldpc(n, d_v, d_c, systematic=True, sparse=True)

            image_coded, image_noisy = ldpc_images.encode_img(G, image_bin, snr)

            #wireless channel
            snr_linear = 10 ** (snr_num / 10)
            var_signal = np.var(image_coded)
            sigma = math.sqrt(var_signal/snr_linear)

            noise = sigma * np.random.randn(*image_coded.shape)
            image_coded_noisy = image_coded + noise


            image_decoded = ldpc_images.decode_img(G, H, image_coded_noisy, snr, image_bin.shape)
            numpy_tensor[i,:,:,:] = image_decoded 

    decoded_images = (numpy_tensor / 127.5) - 1.0
    
    return torch.from_numpy(decoded_images).cuda()










    


