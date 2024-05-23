import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torchvision
import torch.nn.functional as F
from torchvision import datasets, models, transforms
import torch.optim as optim
import os
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np


class ConvAutoencoder(nn.Module):
    def __init__(self, small=None):
        super(ConvAutoencoder, self).__init__()
        ## encoder layers ##
        # conv layer (depth from 1 --> 16), 3x3 kernels
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        # conv layer (depth from 16 --> 4), 3x3 kernels
        self.conv2 = nn.Conv2d(16, 64, 3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        # pooling layer to reduce x-y dims by two; kernel and stride of 2
        self.conv3 = nn.Conv2d(64, 256, 3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 16, 3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(16)
        self.dropout = nn.Dropout(0.5)
        self.fc_en = nn.Linear(16*7*7, 7*7*4)

        self.small = small

        if small is not None:
            self.bn5 = nn.BatchNorm1d(7*7*4)
            self.fc_en_2 = nn.Linear(4*7*7, small)
            self.fc_de = nn.Linear(small, 7 * 7 * 16)
        else:
            self.fc_de = nn.Linear(4 * 7 * 7, 7 * 7 * 16)

        ## decoder layers ##
        ## a kernel of 2 and a stride of 2 will increase the spatial dims by 2
        self.t_conv1 = nn.ConvTranspose2d(16, 16, 2, stride=2)
        self.t_conv2 = nn.ConvTranspose2d(16, 1, 2, stride=2)

    def forward(self, x, output='output'):
        ## encode ##
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))

        x = x.view(x.size(0), -1)
        x = self.dropout(x)                          # mitigate overfitting
        latent = self.fc_en(x)
        if self.small is not None:
            latent = self.bn5(latent)
            latent = self.fc_en_2(latent)

        ## decode ##
        x = self.fc_de(latent)
        x = x.view(x.size(0), 16, 7, 7)
        x = F.relu(self.t_conv1(x))
        x = torch.sigmoid(self.t_conv2(x))

        if output == 'latent':
          return latent
        return x


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        ## encoder layers ##
        # conv layer (depth from 1 --> 16), 3x3 kernels
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        # conv layer (depth from 16 --> 4), 3x3 kernels
        self.conv2 = nn.Conv2d(16, 64, 3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        # pooling layer to reduce x-y dims by two; kernel and stride of 2
        self.conv3 = nn.Conv2d(64, 256, 3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 16, 3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(16)

        self.fc_en = nn.Linear(16*7*7, 7*7*4)

        self.fc_de = nn.Linear(4 * 7 * 7, 10)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        ## encode ##
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))

        x = x.view(x.size(0), -1)

        latent = self.fc_en(x)

        x = self.dropout(latent)
        predict = self.fc_de(F.relu(x))

        return predict


# Configure data loader
os.makedirs("./data/mnist", exist_ok=True)
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST(
        "./data/mnist",
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.Resize(28), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        ),
    ),
    batch_size=64,
    shuffle=True,
)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST(
        "./data/mnist",
        train=False,
        download=True,
        transform=transforms.Compose(
            [transforms.Resize(28), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        ),
    ),
    batch_size=64,
    shuffle=True,
)


def train(featureType='autoencoder'):
    if featureType == 'autoencoder':
        model = ConvAutoencoder()
        criterion = nn.MSELoss()
    else:
        model = Classifier()
        criterion = nn.CrossEntropyLoss()
    model.cuda()

    # specify loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # number of epochs to train the model
    n_epochs = 30

    for epoch in range(1, n_epochs + 1):
        # monitor training loss
        train_loss = 0.0
        test_loss = 0.0
        ###################
        # train the model #
        ###################
        model.train()
        for data in train_loader:
            # _ stands in for labels, here
            # no need to flatten images
            images, labels = data
            images = images.cuda()
            labels = labels.cuda()
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            outputs = model(images)
            # calculate the loss
            if featureType == 'autoencoder':
                loss = criterion(outputs, images)
            else:
                loss = criterion(outputs, labels)
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # update running training loss
            if featureType == 'autoencoder':
                train_loss += loss.item() * images.size(0)
            else:
                train_loss += loss.item()

        # print avg training statistics
        train_loss = train_loss / len(train_loader)

        model.eval()
        correct = 0
        total = 0
        for data in test_loader:
            # _ stands in for labels, here
            # no need to flatten images
            images, labels = data
            images = images.cuda()
            labels = labels.cuda()
            # forward pass: compute predicted outputs by passing inputs to the model
            outputs = model(images)
            # calculate the loss
            if featureType == 'autoencoder':
                loss = criterion(outputs, images)
                test_loss += loss.item() * images.size(0)
            else:
                _, predicted = torch.max(outputs.data, 1)
                # print predicted.type() , t_labels.type()
                total += labels.size(0)
                correct += (predicted == labels).sum()

        test_loss = test_loss / len(test_loader)

        if featureType == 'autoencoder':
            print('Epoch: {} \tTraining Loss: {:.6f} \tTest Loss: {:.6f}'.format(epoch, train_loss, test_loss))
            torch.save(model.state_dict(), "params/AE/AE_v2_%d.pth" % epoch)
        else:
            print('Epoch: {} \tTraining Loss: {:.6f} \tTest Acc: {:.6f}'.format(epoch, train_loss, correct / total))
            torch.save(model.state_dict(), "params/AE/AE_c_%d.pth" % epoch)


def inference():
    # obtain one batch of test images
    model = ConvAutoencoder()
    model.cuda()
    model.load_state_dict(torch.load("params/AE/AE_30.pth"))
    dataiter = iter(test_loader)
    images, labels = dataiter.__next__()
    images = images.cuda()

    output = model(images)
    output = output.cpu()
    # prep images for display
    images = images.cpu().numpy()

    # output is resized into a batch of iages
    output = output.view(64, 1, 28, 28)
    # use detach when it's an output that requires_grad
    output = output.detach().numpy()

    # plot the first ten input images and then reconstructed images
    fig, axes = plt.subplots(nrows=2, ncols=10, sharex=True, sharey=True, figsize=(25, 4))

    # input images on top row, reconstructions on bottom
    for images, row in zip([images, output], axes):
        for img, ax in zip(images, row):
            ax.imshow(np.squeeze(img), cmap='gray')
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
    plt.show()


if __name__ == "__main__":
    train(featureType='autoencoder')  # AE_30.pth or AE_v2_29.pth is the best without or with dropout respectively
