# generate intermediate features

import numpy as np
import pandas as pd
import os, sys
from tqdm import tqdm
import torch

from DP_5_GAN_vfl_LOO_33914 import DiscriminatorClient as D_C_vfl_dp

from GAN_vfl_shadow import DiscriminatorClient as D_C_vfl


batch_size = 42960
cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
torch.manual_seed(0)
np.random.seed(0)

def generate_vfl_gumbel(param_path, data_path, size=42960, delet_target=None):
    df = pd.read_csv('data/adult.csv')

    input = Tensor(df.values)

    if 'DP' in param_path:
        D_C1 = D_C_vfl_dp().cuda()
        D_C2 = D_C_vfl_dp().cuda()
    else:
        D_C1 = D_C_vfl().cuda()
        D_C2 = D_C_vfl().cuda()

    D_C1.load_state_dict(torch.load(param_path + '/D_C1.pth'))
    D_C1.eval()
    D_C2.load_state_dict(torch.load(param_path + '/D_C2.pth'))
    D_C2.eval()

    for i in range(size // batch_size):
        IF_C1 = D_C1(input[:,:53]).cpu().detach().numpy()
        IF_C2 = D_C2(input[:,53:]).cpu().detach().numpy()

    data_dir_1 = data_path + "/IF_C1.npy"
    data_dir_2 = data_path + "/IF_C2.npy"
    np.save(data_dir_1, IF_C1)
    np.save(data_dir_2, IF_C2)


def main(param_path, data_path, size=42960, delet_target=None):
    generate_vfl_gumbel(param_path, data_path, size=size, delet_target=delet_target)


if __name__ == "__main__":
    folders_1 = os.listdir(sys.path[0]+'/params')
    for f_1 in folders_1:
        print(f_1)
        text = f_1.split("_")
        target = text[-1]
        folders_2 = os.listdir(os.path.join(sys.path[0],'params',f_1))
        for f_2 in tqdm(folders_2):
            param_path = os.path.join('params',f_1,f_2)
            data_path = os.path.join('IF_data',f_1,f_2)
            os.makedirs(data_path, exist_ok=True)
            main(param_path, data_path, delet_target=target)
            