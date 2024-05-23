import numpy as np
import pandas as pd
from numpy.random import random
from scipy.linalg import sqrtm
import os
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt

from GAN_vfl_gumbel_mp_shadow import Generator_1 as Generator_1_vfl
from GAN_vfl_gumbel_mp_shadow import Generator_2 as Generator_2_vfl


batch_size = 1599
cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

torch.manual_seed(0)
np.random.seed(0)


def generate_vfl_gumbel(param_path, data_path, size=1599, delet_target=None):
    df = pd.read_csv('data/winequality-red-onehot.csv')
    if delet_target is not None:
        df.drop(index=delet_target, inplace=True)

    G_1 = Generator_1_vfl().cuda()
    span_info = [(5, 11)]
    G_2 = Generator_2_vfl(span_info).cuda()

    G_1.load_state_dict(torch.load(param_path + '/G_1.pth'))
    G_1.eval()
    G_2.load_state_dict(torch.load(param_path + '/G_2.pth'))
    G_2.eval()

    data = np.zeros([size, 17])

    for i in range(size // batch_size):
        noise = Tensor(np.random.normal(0, 1, (batch_size, 10)))
        fake_imgs_client_1 = G_1(noise)
        fake_imgs_client_2 = G_2(noise)
        fake_data = torch.cat((fake_imgs_client_1, fake_imgs_client_2), dim=1)
        data[batch_size * i:batch_size * (i + 1), :] = fake_data.cpu().detach().numpy()

    for i in range(11):
        mean = df.iloc[:, i].mean()
        std = df.iloc[:, i].std()
        data[:, i] = (data[:, i]*std) + mean
    
    data_dir = data_path + "/data_onehot.npy"
    np.save(data_dir, data)
    data[:, 11] = data[:, 11] + data[:, 12]*2 + data[:, 13]*3 + data[:, 14]*4 + data[:, 15]*5 + data[:, 16]*6 + 2
    data_dir_2 = data_path + "/data.npy"
    np.save(data_dir_2, data[:, :12])
    

def main(param_path, data_path, size=1599, mode='centric', delet_target=None):
    generate_vfl_gumbel(param_path, data_path, size=size, delet_target=delet_target)
   

if __name__ == "__main__":
    for i in tqdm(range(100)):
        param_path = "params/DP_WGAN_vfl_LOO_epsilon_5_151/" + str(i)
        data_path = "generated_data/DP_WGAN_vfl_LOO_epsilon_5_151/" + str(i)
        os.makedirs(data_path, exist_ok=True)
        main(param_path, data_path, delet_target=151)
