# generate intermediate features

import numpy as np
import pandas as pd
import os, sys
from tqdm import tqdm
import torch

from DP_5_GAN_vfl_mp_LOO_2781 import DiscriminatorClient_1 as D_C1_vfl_dp
from DP_5_GAN_vfl_mp_LOO_2781 import DiscriminatorClient_2 as D_C2_vfl_dp

from GAN_vfl_gumbel_mp_shadow import DiscriminatorClient_1 as D_C1_vfl
from GAN_vfl_gumbel_mp_shadow import DiscriminatorClient_2 as D_C2_vfl


batch_size = 4898
cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
torch.manual_seed(0)
np.random.seed(0)

def generate_vfl_gumbel(param_path, data_path, size=4898, delet_target=None):
    df = pd.read_csv('data/winequality-white-onehot.csv')
    # if delet_target.isnumeric():
    #     idx = int(delet_target)
    #     df.drop(index=idx, inplace=True)

    input = Tensor(df.values)

    if 'DP' in param_path:
        D_C1 = D_C1_vfl_dp().cuda()
        D_C2 = D_C2_vfl_dp().cuda()
    else:
        D_C1 = D_C1_vfl().cuda()
        D_C2 = D_C2_vfl().cuda()

    # print(param_path)

    D_C1.load_state_dict(torch.load(param_path + '/D_C1.pth'))
    D_C1.eval()
    D_C2.load_state_dict(torch.load(param_path + '/D_C2.pth'))
    D_C2.eval()

    # IF_C1 = np.zeros([size, 32])
    # IF_C2 = np.zeros([size, 32])

    for i in range(size // batch_size):
        IF_C1 = D_C1(input[:,:6]).cpu().detach().numpy()
        IF_C2 = D_C2(input[:,6:]).cpu().detach().numpy()

    data_dir_1 = data_path + "/IF_C1.npy"
    data_dir_2 = data_path + "/IF_C2.npy"
    np.save(data_dir_1, IF_C1)
    np.save(data_dir_2, IF_C2)


def main(param_path, data_path, size=4898, delet_target=None):
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
            