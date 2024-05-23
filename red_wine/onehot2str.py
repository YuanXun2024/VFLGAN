import numpy as np
import pandas as pd
from tqdm import tqdm


def main(data_path):
    data = np.load(data_path+'data.npy')
    df = pd.read_csv('./data/winequality-red.csv')
    df_np = pd.DataFrame(data)
    df_np.columns = df.columns

    df_np.to_csv(data_path+'redwine.csv')


if __name__ == '__main__':
    for i in tqdm(range(100)):
        data_path = "generated_data/DP_WGAN_vfl_LOO_epsilon_5_1235/" + str(i) + '/'
        main(data_path)



