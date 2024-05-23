import numpy
from numpy import cov
from numpy import trace
from numpy import iscomplexobj
from numpy.random import random
from scipy.linalg import sqrtm
import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import sys
from AutoEncoder import ConvAutoencoder, Classifier
from DP_WGAN_vfl import Generator
from WGAN_GP_real_VFL_v2 import Generator as Generator_v2


batch_size = 30000
cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


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
    batch_size=batch_size,
    shuffle=True,
)


def calculate_fid(act1, act2):
    # calculate mean and covariance statistics
    mu1, sigma1 = act1.mean(axis=0), cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), cov(act2, rowvar=False)
    # calculate sum squared difference between means
    ssdiff = numpy.sum((mu1 - mu2)**2.0)
    # calculate sqrt of product between cov
    covmean = sqrtm(sigma1.dot(sigma2))
    # check and correct imaginary numbers from sqrt
    if iscomplexobj(covmean):
        covmean = covmean.real
    # calculate score
    fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid


def featureReal(modelType ='AE'):
    if modelType == 'AE':
        AE = ConvAutoencoder().cuda()
        AE.load_state_dict(torch.load("params/AE/AE_30.pth"))
    else:
        AE = Classifier().cuda()
        AE.load_state_dict(torch.load("params/AE/AE_c_19.pth"))
    AE.eval()
    act1 = numpy.zeros([60000, 196])

    for i, (imgs, _) in enumerate(train_loader):
        images = imgs.cuda()
        latent = AE(images, output='latent')
        latent = latent.cpu().detach().numpy()
        act1[batch_size*i:batch_size*(i+1), :] = latent

    return act1


def featureGenerativeVFL(epoch, version=1):
    AE = ConvAutoencoder().cuda()

    if version == 1:
        G_1 = Generator().cuda()
        G_2 = Generator().cuda()
    elif version ==2:
        G_1 = Generator_v2().cuda()
        G_2 = Generator_v2().cuda()
    else:
        G_1 = Generator_v2().cuda()
        G_2 = Generator_v2().cuda()

    AE.load_state_dict(torch.load("params/AE/AE_30.pth"))
    AE.eval()

    name_1 = 'G_1_' + str(epoch) + '.pth'
    name_2 = 'G_2_' + str(epoch) + '.pth'

    if version == 1:
        G_1.load_state_dict(torch.load("params/DP_vfl/"+name_1))
        G_2.load_state_dict(torch.load("params/DP_vfl/"+name_2))
    elif version == 2:
        G_1.load_state_dict(torch.load("params/WGAN_GP_real_VFL_v2_40/" + name_1))
        G_2.load_state_dict(torch.load("params/WGAN_GP_real_VFL_v2_40/" + name_2))
    else:
        G_1.load_state_dict(torch.load("params/Adversary_target_0/" + name_1))
        G_2.load_state_dict(torch.load("params/Adversary_target_0/" + name_2))
    G_1.eval()
    G_2.eval()

    act2 = numpy.zeros([60000, 196])

    for i in range(60000//batch_size):

        noise = Tensor(numpy.random.normal(0, 1, (batch_size, 100)))

        fake_imgs_client_1 = G_1(noise)
        fake_imgs_client_2 = G_2(noise)
        fake_imgs = torch.cat((fake_imgs_client_1, fake_imgs_client_2), dim=2)
        latent = AE(fake_imgs, output='latent')
        latent = latent.cpu().detach().numpy()
        act2[batch_size * i:batch_size * (i + 1), :] = latent

    return act2  


if __name__ == "__main__":
    act1 = featureReal(modelType='AE')
    GenerativeModelList = ['vfl']
    ours = []; DPSGD = []; GS_WGAN = []; vfl=[];
    for model in GenerativeModelList:
        for i in range(1, 31):
            if model == 'vfl':
                act2 = featureGenerativeVFL(epoch=10*i)
                fid = calculate_fid(act1, act2)
                vfl += [fid]    

    print('vfl min fid: ', min(vfl), vfl.index(min(vfl)))

    fig = plt.figure()
    plt.xlabel('epoch')
    plt.ylabel('FD score')
    x_axis = [10*i for i in range(1, 31)]
    for model in GenerativeModelList:
        if model == 'vfl':
            plt.plot(x_axis, vfl, '*-', label='DP-VFLGAN')

    plt.legend()
    plt.savefig('FD_dp.png', bbox_inches='tight')
