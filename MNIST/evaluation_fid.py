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


from AutoEncoder import ConvAutoencoder, Classifier
from WGAN_GP import Generator as Generator_centric
from WGAN_GP_VFL import GeneratorBackbone, GeneratorHead
from WGAN_GP_real_VFL import Generator
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


def featureGenerativeHFL(epoch):
    AE = ConvAutoencoder().cuda()
    G_backbone = GeneratorBackbone().cuda()
    G_head_1 = GeneratorHead().cuda()
    G_head_2 = GeneratorHead().cuda()

    AE.load_state_dict(torch.load("params/AE/AE_30.pth"))
    AE.eval()

    backbone = 'backbone_' + str(epoch) + '.pth'
    head_client1 = 'head_client1_' + str(epoch) + '.pth'
    head_client2 = 'head_client2_' + str(epoch) + '.pth'

    G_backbone.load_state_dict(torch.load("params/WGAN_GP_VFL/"+backbone))
    G_head_1.load_state_dict(torch.load("params/WGAN_GP_VFL/"+head_client1))
    G_head_2.load_state_dict(torch.load("params/WGAN_GP_VFL/"+head_client2))
    G_backbone.eval()
    G_head_1.eval()
    G_head_2.eval()

    act2 = numpy.zeros([60000, 196])

    for i in range(60000//batch_size):

        noise = Tensor(numpy.random.normal(0, 1, (batch_size, 100)))

        fake_imgs_client_1 = G_head_1(G_backbone(noise))
        fake_imgs_client_2 = G_head_2(G_backbone(noise))
        fake_imgs = torch.cat((fake_imgs_client_1, fake_imgs_client_2), dim=2)
        latent = AE(fake_imgs, output='latent')
        latent = latent.cpu().detach().numpy()
        act2[batch_size * i:batch_size * (i + 1), :] = latent

    return act2


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
        G_1.load_state_dict(torch.load("params/WGAN_GP_real_VFL/"+name_1))
        G_2.load_state_dict(torch.load("params/WGAN_GP_real_VFL/"+name_2))
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


def featureGenerativeCentric(epoch):
    AE = ConvAutoencoder().cuda()
    G = Generator_centric().cuda()

    AE.load_state_dict(torch.load("params/AE/AE_30.pth"))
    AE.eval()

    name = 'generator_' + str(epoch) + '.pth'

    G.load_state_dict(torch.load("params/WGAN_GP/" + name))
    G.eval()

    act2 = numpy.zeros([60000, 196])

    for i in range(60000//batch_size):

        noise = Tensor(numpy.random.normal(0, 1, (batch_size, 100)))
        fake_imgs = G(noise)
        latent = AE(fake_imgs, output='latent')
        latent = latent.cpu().detach().numpy()
        act2[batch_size * i:batch_size * (i + 1), :] = latent

    return act2


if __name__ == "__main__":
    act1 = featureReal(modelType='AE')
    GenerativeModelList = ['centric', 'HFL', 'VFL', 'VFL_v2']
    # GenerativeModelList = ['VFL_v2']
    fid_c = []; fid_hfl = []; fid_vfl = []; fid_vfl_v2 = []; fid_adv = []
    for model in GenerativeModelList:
        for i in range(1, 61):
            if model == 'centric':
                act2 = featureGenerativeCentric(epoch=5*i)
                fid = calculate_fid(act1, act2)
                fid_c += [fid]
            elif model == 'HFL':
                act2 = featureGenerativeHFL(epoch=5 * i)
                fid = calculate_fid(act1, act2)
                fid_hfl += [fid]
            elif model == 'VFL':
                act2 = featureGenerativeVFL(epoch=5 * i)
                fid = calculate_fid(act1, act2)
                fid_vfl += [fid]
            elif model == 'VFL_v2':
                act2 = featureGenerativeVFL(epoch=5 * i, version=2)
                fid = calculate_fid(act1, act2)
                fid_vfl_v2 += [fid]
                # act2 = featureGenerativeVFL(epoch=5 * i, version=3)
                # fid = calculate_fid(act1, act2)
                # fid_adv += [fid]

    print('WGAN_GP min fid: ', min(fid_c), fid_c.index(min(fid_c)))
    print('VertiGAN min fid: ', min(fid_hfl), fid_hfl.index(min(fid_hfl)))
    print('VFLGAN-base min fid: ', min(fid_vfl), fid_vfl.index(min(fid_vfl)))
    print('VFLGAN min fid: ', min(fid_vfl_v2), fid_vfl_v2.index(min(fid_vfl_v2)))
    # print('adv min fid: ', min(fid_adv), fid_adv.index(min(fid_adv)))

    fig = plt.figure()
    plt.xlabel('epoch')
    plt.ylabel('FD score')
    x_axis = [5*i for i in range(1, 61)]
    for model in GenerativeModelList:
        if model == 'centric':
            plt.plot(x_axis, fid_c, '*-', label='WGAN_GP')
        elif model == 'HFL':
            plt.plot(x_axis, fid_hfl, '*-', label='VertiGAN')
        elif model == 'VFL':
            plt.plot(x_axis, fid_vfl, '*-', label='VFLGAN-base')
        elif model == 'VFL_v2':
            plt.plot(x_axis, fid_vfl_v2, '*-', label='VFLGAN')
    plt.legend()
    plt.savefig('FD.png', bbox_inches='tight')
    #plt.show()
