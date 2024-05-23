import numpy as np
import pandas as pd
from tqdm import tqdm


def main(data_path):
    data = np.load(data_path+'data.npy')
    df = pd.read_csv('./data/adult.csv')
    df_np = pd.DataFrame(data)
    df_np.columns = df.columns

    df_np['workclass'] = '?'
    df_np['education'] = '?'
    df_np['marital-status'] = '?'
    df_np['occupation'] = '?'
    df_np['relationship'] = '?'
    df_np['race'] = '?'
    df_np['sex'] = '?'
    df_np['native-country'] = '?'
    df_np['income'] = '?'

    for col in df.columns:
        if col.find('workclass')>=0:
            df_np.loc[df_np[col] == 1, 'workclass'] = col.replace('workclass_', '')
        elif col.find('education')>=0:
            df_np.loc[df_np[col] == 1, 'education'] = col.replace('education_', '')
        elif col.find('marital-status') >= 0:
            df_np.loc[df_np[col] == 1, 'marital-status'] = col.replace('marital-status_', '')
        elif col.find('occupation')>=0:
            df_np.loc[df_np[col] == 1, 'occupation'] = col.replace('occupation_', '')
        elif col.find('relationship')>=0:
            df_np.loc[df_np[col] == 1, 'relationship'] = col.replace('relationship_', '')
        elif col.find('race')>=0:
            df_np.loc[df_np[col] == 1, 'race'] = col.replace('race_', '')
        elif col.find('sex')>=0:
            df_np.loc[df_np[col] == 1, 'sex'] = col.replace('sex_', '')
        elif col.find('native-country')>=0:
            df_np.loc[df_np[col] == 1, 'native-country'] = col.replace('native-country_', '')
        elif col.find('income')>=0:
            df_np.loc[df_np[col] == 1, 'income'] = col.replace('income_', '')

    df_np.to_csv(data_path+'adult.csv')


if __name__ == '__main__':
    for i in tqdm(range(100)):
        data_path = "generated_data/WGAN_vfl_LOO_37592/" + str(i) + '/'
        main(data_path)


