import os
import gzip
import numpy as np
from torch.utils.data import Dataset


def load_data(data_folder, data_name, label_name):
    with gzip.open(os.path.join(data_folder, label_name), 'rb') as lbpath:
        y_train = np.frombuffer(lbpath.read(), np.uint8, offset=8)

    with gzip.open(os.path.join(data_folder, data_name), 'rb') as imgpath:
        x_train = np.frombuffer(
            imgpath.read(), np.uint8, offset=16).reshape(len(y_train), 28, 28)
    return x_train, y_train


class CustomDataset(Dataset):
    def __init__(self, folder, data_name, label_name, targetRecord=0, transform=None):
        (train_set, train_labels) = load_data(folder, data_name, label_name)
        self.train_set = np.delete(train_set, targetRecord, axis=0)
        self.train_labels = np.delete(train_labels, targetRecord)
        self.transform = transform

    def __getitem__(self, index):
        img, target = np.array(self.train_set[index]), int(self.train_labels[index])
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.train_set)
    