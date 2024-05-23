import numpy as np
import pandas as pd
from tqdm import tqdm
import os, sys


def main(data_path):
    data = np.load(data_path+'/data.npy')
    df = pd.read_csv('./data/winequality-white.csv')
    df_np = pd.DataFrame(data)
    df_np.columns = df.columns

    df_np.to_csv(data_path+'/whitewine.csv')


# if __name__ == '__main__':
#     for i in tqdm(range(100)):
#         data_path = "generated_data/DP_WGAN_vfl_shadow_epsilon_5/" + str(i) + '/'
#         main(data_path)

if __name__ == "__main__":
    folders_1 = os.listdir(sys.path[0]+'/generated_data')
    for f_1 in folders_1:
        print(f_1)
        text = f_1.split("_")
        target = text[-1]
        folders_2 = os.listdir(os.path.join(sys.path[0],'params',f_1))
        for f_2 in tqdm(folders_2):
            data_path = os.path.join('generated_data',f_1,f_2)
            main(data_path)