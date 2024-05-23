import numpy as np
from numpy.random.mtrand import noncentral_chisquare
import pandas as pd
from numpy.random import random
from scipy.linalg import sqrtm
import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
from tqdm import tqdm

from GAN_vfl_shadow import Generator as Generator_vfl


batch_size = 42960
cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

torch.manual_seed(0)
np.random.seed(0)


def generate_vfl(param_path, data_path, size=42960, delet_target=None):
    span_info_1 = [(0, 0+1), (1, 1+7), (8, 8+1), (9, 9+16), (25, 25+1), (26, 26+7), (33, 33+14), (47, 47+6)]
    span_info_2 = [(0, 0+5), (58-53, 58+2-53), (60-53, 60+1-53), (61-53, 61+1-53), (62-53, 62+1-53), (63-53, 63+41-53), (104-53, 104+2-53)]

    G_1 = Generator_vfl(span_info_1).cuda()
    G_2 = Generator_vfl(span_info_2).cuda()

    G_1.load_state_dict(torch.load(param_path + '/G_1.pth'))
    G_1.eval()
    G_2.load_state_dict(torch.load(param_path + '/G_2.pth'))
    G_2.eval()

    data = np.zeros([size, 106])

    for i in range(size // batch_size):
        noise = Tensor(np.random.normal(0, 1, (batch_size, 20)))
        fake_imgs_client_1 = G_1(noise)
        fake_imgs_client_2 = G_2(noise)
        fake_data = torch.cat((fake_imgs_client_1, fake_imgs_client_2), dim=1)
        data[batch_size * i:batch_size * (i + 1), :] = fake_data.cpu().detach().numpy()
    
    data_new = np.zeros_like(data)

    # data stored in client 1
    data_new[:, 0] = data[:, 0]
    data_new[:, 6:6+7] = data[:, 1:1+7]
    data_new[:, 1] = data[:, 8]
    data_new[:, 13:13+16] = data[:, 9:9+16]
    data_new[:, 2] = data[:, 25]
    data_new[:, 29:29+7] = data[:, 26:26+7]
    data_new[:, 36:36+14] = data[:, 33:33+14]
    data_new[:, 47:47+6] = data[:, 50:50+6]

    # data stored in client 2
    data_new[:, 56:56+5] = data[:, 53:53+5]
    data_new[:, 61:61+2] = data[:, 58:58+2]
    data_new[:, 3] = data[:, 60]
    data_new[:, 4] = data[:, 61]
    data_new[:, 5] = data[:, 62]
    data_new[:, 63:63+41] = data[:, 63:63+41]
    data_new[:, 104:104+2] = data[:, 104:104+2]
    df = pd.read_csv('data/adult.csv')
    if delet_target is not None:
        df.drop(index=delet_target, inplace=True)

    for i in range(6):
        mean = df.iloc[:, i].mean()
        std = df.iloc[:, i].std()
        data_new[:, i] = (data_new[:, i]*std) + mean

    data_dir = data_path + "/data.npy"
    np.save(data_dir, data_new)


def main(param_path, data_path, size=42960, mode='centric', delet_target=None):
    generate_vfl(param_path, data_path, size=size, delet_target=delet_target)


if __name__ == "__main__":
    for i in tqdm(range(100)):
        param_path = "params/WGAN_vfl_shadow_non_epsilon/" + str(i)
        data_path = "generated_data/WGAN_vfl_shadow_non_epsilon/" + str(i)
        os.makedirs(data_path, exist_ok=True)
        main(param_path, data_path, mode='vfl', delet_target=None)