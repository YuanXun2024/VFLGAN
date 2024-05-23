import pandas as pd
import numpy as np
from scipy.linalg import sqrtm
import copy


def calculate_fid(act1, act2):
    # calculate mean and covariance statistics
    mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)
    # calculate sum squared difference between means
    ssdiff = np.sum((mu1 - mu2)**2.0)
    # calculate sqrt of product between cov
    covmean = sqrtm(sigma1.dot(sigma2))
    # check and correct imaginary numbers from sqrt
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    # calculate score
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid


import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score, ShuffleSplit
from sklearn.metrics import accuracy_score, f1_score


def train_x_test_x(model, data_path, norm=True, feature_num=11, n_splits=10):
    filename, type = os.path.splitext(data_path)
    if type == '.csv':
        df = pd.read_csv(data_path)
        data = df.values
    elif type == '.npy':
        data = np.load(data_path)
        data = data[:1599]
    else:
        raise NotImplementedError

    x = data[:, :feature_num]
    y = data[:, feature_num]
    if norm:
        mean = x.mean(axis=0, keepdims=True)
        std = x.std(axis=0, keepdims=True)
        x = (x-mean)/std

    cv = ShuffleSplit(n_splits=n_splits, test_size=0.3, random_state=0)
    scores = cross_val_score(model, x, y, cv=cv)
    f1 = cross_val_score(model, x, y, cv=cv, scoring="f1_macro")

    print("Accuracy:%.2f(+/- %.2f)" % (scores.mean(), scores.std() * 2))
    print("f1 score:%.2f(+/- %.2f)" % (f1.mean(), f1.std() * 2))


def train_x_test_y(model, data_path_1, data_path_2, norm=True, feature_num=11, n_splits=10):
    filename, type = os.path.splitext(data_path_1)
    if type == '.csv':
        df = pd.read_csv(data_path_1)
        data_1 = df.values
    elif type == '.npy':
        data_1 = np.load(data_path_1)
        data_1 = data_1[:1599]
    else:
        raise NotImplementedError

    filename, type = os.path.splitext(data_path_2)
    if type == '.csv':
        df = pd.read_csv(data_path_2)
        data_2 = df.values
    elif type == '.npy':
        data_2 = np.load(data_path_2)
        data_2 = data_2[:1599]
    else:
        raise NotImplementedError

    x_train = data_1[:, :feature_num]
    y_train = data_1[:, feature_num]
    x_test = data_2[:, :feature_num]
    y_test = data_2[:, feature_num]
    if norm:
        mean = x_train.mean(axis=0, keepdims=True)
        std = x_train.std(axis=0, keepdims=True)
        x_train = (x_train-mean)/std
        mean = x_test.mean(axis=0, keepdims=True)
        std = x_test.std(axis=0, keepdims=True)
        x_test = (x_test - mean) / std

    model.fit(x_train, y_train)

    pre_test = model.predict(x_test)

    print("Accuracy: %.2f" % accuracy_score(y_test, pre_test))
    print("f1 score: %.2f" % f1_score(y_test, pre_test, average='macro'))


def main():
    real_data_path = 'winequality-red.csv'
    fake_data_path_dlpt = 'red_wine_dlpt.csv'
    
    model = RandomForestClassifier(bootstrap=True, ccp_alpha=0.0,
                          criterion='gini', max_depth=3, max_features=1.0,
                          min_impurity_decrease=0.0,
                          min_samples_leaf=1, min_samples_split=2,
                          min_weight_fraction_leaf=0.0, n_estimators=100,
                          n_jobs=None, oob_score=False, random_state=0,
                          verbose=0, warm_start=False)
    # model = MLPClassifier(solver='sgd', activation='relu', alpha=1e-4, hidden_layer_sizes=(64, 32), random_state=1,
    #                 max_iter=1000, verbose=False, learning_rate_init=.1)
    print('real data')
    print('train on real test on real:')
    train_x_test_x(model, real_data_path)

    print('DLPT')
    print('train on fake test on fake:')
    train_x_test_x(model, fake_data_path_dlpt)
    print('train on real test on fake:')
    train_x_test_y(model, real_data_path, fake_data_path_dlpt)
    print('train on fake test on real:')
    train_x_test_y(model, fake_data_path_dlpt, real_data_path)


if __name__ == '__main__':
    data_real = pd.read_csv('winequality-red-onehot.csv')
    data_dlpt = pd.read_csv('red_wine_dlpt.csv')

    data_real = data_real.values
    data_dlpt = data_dlpt.values

    data_syn = np.zeros_like(data_real)
    data_syn[:, :11] = data_dlpt[:, :11]

    for row in range(data_syn.shape[0]):
        tmp = np.zeros([6])
        idx = data_dlpt[row, 11] - 3
        tmp[int(idx)] = 1
        data_syn[row, 11:] = tmp

    data_real_norm = copy.deepcopy(data_real)
    data_dlpt_norm = copy.deepcopy(data_syn)

    print(data_dlpt_norm.shape, data_real_norm.shape)

    for i in range(11):
        data_real_norm[:, i] = (data_real_norm[:, i]-data_real_norm[:, i].mean()) / data_real_norm[:, i].std()
        data_dlpt_norm[:, i] = (data_dlpt_norm[:, i]-data_real_norm[:, i].mean()) / data_real_norm[:, i].std()

    fid = calculate_fid(data_real_norm, data_dlpt_norm)
    print('FID:', fid)
    main()