import torch
import pandas as pd
from pandas import DataFrame
import numpy as np
import random
import json

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score, ShuffleSplit

from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

from features import Features
from independent_histograms import NaiveFeatureSet

random.seed(0)
torch.manual_seed(0)
np.random.seed(0)

batch_size = 1599
cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


class MIAttackClassifier:
    def __init__(self, model, feature="corr"):
        self.model = model
        self.feature = feature

        self.trained = False

    def train(self, data, labels, n_splits=10):
        cv = ShuffleSplit(n_splits=n_splits, test_size=0.3, random_state=0)
        scores = cross_val_score(self.model, data, labels, cv=cv)

        return scores


def main():
    shadow_path = "generated_data/WGAN_vfl_shadow_non_epsilon/"
    ano_path = "generated_data/WGAN_vfl_LOO_33914/"
    
    # shadow_path = "generated_data/DP_WGAN_vfl_shadow_epsilon_5/"
    # ano_path = "generated_data/DP_WGAN_vfl_LOO_epsilon_5_1235/"
    print(shadow_path, ano_path)


    order_list = [i for i in range(100)]
    scores = []
    
    F = NaiveFeatureSet(DataFrame)

    for _ in range(5):
        feature_shadow = []
        feature_ano = []
        random.shuffle(order_list)
        train_set = order_list[:70]
        # print(train_set)
        test_set = order_list[70:]
        # print(test_set)
        for i in train_set:
            shadow_dir = shadow_path + str(i) + "/adult.csv"
            data = pd.read_csv(shadow_dir)
            ano_dir = ano_path + str(i) + "/adult.csv"
            data_ano = pd.read_csv(ano_dir)
            for j in range(1):     
                feature_shadow += [F.extract(data)]          
                feature_ano += [F.extract(data_ano)]

        features = feature_shadow + feature_ano
        features = np.array(features)
        print(features.shape)
        labels = np.zeros(features.shape[0])
        labels[features.shape[0]//2:] = 1

        model = RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, 
                          criterion='gini', max_depth=3, max_features='sqrt',
                          min_impurity_decrease=0.0,
                          min_samples_leaf=1, min_samples_split=2,
                          min_weight_fraction_leaf=0.0, n_estimators=1000,
                          n_jobs=None, oob_score=False, random_state=0,
                          verbose=0, warm_start=False)
        # model = MLPClassifier(solver='sgd', activation='relu', alpha=1e-4, 
        #                       hidden_layer_sizes=(24, 10), random_state=0,
        #                       max_iter=1000, verbose=False, learning_rate_init=.1)
                          
        model = model.fit(features, labels)

        feature_shadow = []
        feature_ano = []

        for i in test_set:
            shadow_dir = shadow_path + str(i) + "/adult.csv"
            data = pd.read_csv(shadow_dir)
            ano_dir = ano_path + str(i) + "/adult.csv"
            data_ano = pd.read_csv(ano_dir)
            for j in range(1):  
                feature_shadow += [F.extract(data)]          
                feature_ano += [F.extract(data_ano)]
        
        features = feature_shadow + feature_ano
        features = np.array(features)
        print(features.shape)
        labels = np.zeros(features.shape[0])
        labels[features.shape[0]//2:] = 1

        scores += [model.score(features, labels)]
    scores = np.array(scores)
    print("Accuracy:%.2f(+/- %.2f)" % (scores.mean(), scores.std() * 2))
    print(scores)


if __name__ == "__main__":
    main()
