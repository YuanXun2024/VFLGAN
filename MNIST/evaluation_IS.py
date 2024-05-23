import numpy
import numpy as np
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

import random
random.seed(19950522)

torch.manual_seed(19950522)
numpy.random.seed(19950522)

batch_size = 15000
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


eval_size = 60000
class_num = 10


def calculate_IS(eps=1E-16, model=None, epoch=0):
    classifier = Classifier().cuda()
    classifier.load_state_dict(torch.load("params/AE/AE_c_19.pth"))
    classifier.eval()

    p_yx_all = numpy.zeros([eval_size, class_num])

    if model == 'centric':
        G = Generator_centric().cuda()
        name = 'generator_' + str(epoch) + '.pth'
        G.load_state_dict(torch.load("params/WGAN_GP/" + name))
        G.eval()
        for i in range(eval_size // batch_size):
            noise = Tensor(numpy.random.normal(0, 1, (batch_size, 100)))
            fake_imgs = G(noise)
            predict = classifier(fake_imgs)
            p_yx = torch.nn.functional.softmax(predict, dim=1)
            p_yx = p_yx.cpu().detach().numpy()
            p_yx_all[batch_size * i:batch_size * (i + 1), :] = p_yx
        p_y_all = numpy.expand_dims(p_yx_all.mean(axis=0), 0)
        kl_d = p_yx_all * (numpy.log(p_yx_all+eps)-numpy.log(p_y_all+eps))
        sum_kl_d = kl_d.sum(axis=1)
        avg_kl_d = sum_kl_d.mean()
        is_score = np.exp(avg_kl_d)
    elif model == 'HFL':
        G_backbone = GeneratorBackbone().cuda()
        G_head_1 = GeneratorHead().cuda()
        G_head_2 = GeneratorHead().cuda()
        backbone = 'backbone_' + str(epoch) + '.pth'
        head_client1 = 'head_client1_' + str(epoch) + '.pth'
        head_client2 = 'head_client2_' + str(epoch) + '.pth'
        G_backbone.load_state_dict(torch.load("params/WGAN_GP_VFL/" + backbone))
        G_head_1.load_state_dict(torch.load("params/WGAN_GP_VFL/" + head_client1))
        G_head_2.load_state_dict(torch.load("params/WGAN_GP_VFL/" + head_client2))
        G_backbone.eval()
        G_head_1.eval()
        G_head_2.eval()
        for i in range(eval_size // batch_size):
            noise = Tensor(numpy.random.normal(0, 1, (batch_size, 100)))

            fake_imgs_client_1 = G_head_1(G_backbone(noise))
            fake_imgs_client_2 = G_head_2(G_backbone(noise))
            fake_imgs = torch.cat((fake_imgs_client_1, fake_imgs_client_2), dim=2)
            predict = classifier(fake_imgs)
            p_yx = torch.nn.functional.softmax(predict, dim=1)
            p_yx = p_yx.cpu().detach().numpy()
            p_yx_all[batch_size * i:batch_size * (i + 1), :] = p_yx
        p_y_all = numpy.expand_dims(p_yx_all.mean(axis=0), 0)
        kl_d = p_yx_all * (numpy.log(p_yx_all+eps)-numpy.log(p_y_all+eps))
        sum_kl_d = kl_d.sum(axis=1)
        avg_kl_d = sum_kl_d.mean()
        is_score = np.exp(avg_kl_d)
    elif model == 'VFL':
        G_1 = Generator().cuda()
        G_2 = Generator().cuda()
        name_1 = 'G_1_' + str(epoch) + '.pth'
        name_2 = 'G_2_' + str(epoch) + '.pth'
        G_1.load_state_dict(torch.load("params/WGAN_GP_real_VFL/" + name_1))
        G_2.load_state_dict(torch.load("params/WGAN_GP_real_VFL/" + name_2))
        G_1.eval()
        G_2.eval()
        for i in range(eval_size // batch_size):
            noise = Tensor(numpy.random.normal(0, 1, (batch_size, 100)))

            fake_imgs_client_1 = G_1(noise)
            fake_imgs_client_2 = G_2(noise)
            fake_imgs = torch.cat((fake_imgs_client_1, fake_imgs_client_2), dim=2)
            predict = classifier(fake_imgs)
            p_yx = torch.nn.functional.softmax(predict, dim=1)
            p_yx = p_yx.cpu().detach().numpy()
            p_yx_all[batch_size * i:batch_size * (i + 1), :] = p_yx
        p_y_all = numpy.expand_dims(p_yx_all.mean(axis=0), 0)
        kl_d = p_yx_all * (numpy.log(p_yx_all+eps)-numpy.log(p_y_all+eps))
        sum_kl_d = kl_d.sum(axis=1)
        avg_kl_d = sum_kl_d.mean()
        is_score = np.exp(avg_kl_d)
    elif model == 'VFL_v2':
        G_1 = Generator_v2().cuda()
        G_2 = Generator_v2().cuda()
        name_1 = 'G_1_' + str(epoch) + '.pth'
        name_2 = 'G_2_' + str(epoch) + '.pth'
        G_1.load_state_dict(torch.load("params/WGAN_GP_real_VFL_v2_40/" + name_1))
        G_2.load_state_dict(torch.load("params/WGAN_GP_real_VFL_v2_40/" + name_2))
        G_1.eval()
        G_2.eval()
        for i in range(eval_size // batch_size):
            noise = Tensor(numpy.random.normal(0, 1, (batch_size, 100)))

            fake_imgs_client_1 = G_1(noise)
            fake_imgs_client_2 = G_2(noise)
            fake_imgs = torch.cat((fake_imgs_client_1, fake_imgs_client_2), dim=2)
            predict = classifier(fake_imgs)
            p_yx = torch.nn.functional.softmax(predict, dim=1)
            p_yx = p_yx.cpu().detach().numpy()
            p_yx_all[batch_size * i:batch_size * (i + 1), :] = p_yx
        p_y_all = numpy.expand_dims(p_yx_all.mean(axis=0), 0)
        kl_d = p_yx_all * (numpy.log(p_yx_all+eps)-numpy.log(p_y_all+eps))
        sum_kl_d = kl_d.sum(axis=1)
        avg_kl_d = sum_kl_d.mean()
        is_score = np.exp(avg_kl_d)
    else:
        for i, (imgs, _) in enumerate(train_loader):
            # if i == 10:
            #     break
            images = imgs.cuda()
            predict = classifier(images)
            p_yx = torch.nn.functional.softmax(predict, dim=1)
            p_yx = p_yx.cpu().detach().numpy()
            p_yx_all[batch_size * i:batch_size * (i + 1), :] = p_yx
        p_y_all = numpy.expand_dims(p_yx_all.mean(axis=0), 0)
        kl_d = p_yx_all * (numpy.log(p_yx_all + eps) - numpy.log(p_y_all + eps))
        sum_kl_d = kl_d.sum(axis=1)
        avg_kl_d = sum_kl_d.mean()
        is_score = np.exp(avg_kl_d)

    return is_score


if __name__ == "__main__":

    GenerativeModelList = ['centric', 'HFL', 'VFL', 'VFL_v2', 'real']

    is_real = []; is_c = []; is_hfl = []; is_vfl = []; is_vfl_v2 = []
    for model in GenerativeModelList:
        for i in range(1, 61):
            if model == 'centric':
                is_score = calculate_IS(model=model, epoch=5*i)
                is_c += [is_score]
            elif model == 'HFL':
                is_score = calculate_IS(model=model, epoch=5 * i)
                is_hfl += [is_score]
            elif model == 'VFL':
                is_score = calculate_IS(model=model, epoch=5 * i)
                is_vfl += [is_score]
            elif model == 'VFL_v2':
                is_score = calculate_IS(model=model, epoch=5*i)
                is_vfl_v2 += [is_score]
            else:
                break
        if model == 'real':
            is_score = calculate_IS()
            is_real = [is_score for i in range(60)]

    print('centric max IS: ', max(is_c))
    print('HFL max IS: ', max(is_hfl))
    print('vfl max IS: ', max(is_vfl))
    print('vfl_v2 max IS: ', max(is_vfl_v2))
    print('real images max IS: ', max(is_real))

    fig = plt.figure()
    plt.xlabel('epoch')
    plt.ylabel('IS score')
    x_axis = [5*i for i in range(1, 61)]
    for model in GenerativeModelList:
        if model == 'centric':
            plt.plot(x_axis, is_c, '*-', label='WGAN_GP')
        elif model == 'HFL':
            plt.plot(x_axis, is_hfl, '*-', label='VertiGAN')
        elif model == 'VFL':
            plt.plot(x_axis, is_vfl, '*-', label='VFLGAN-base')
        elif model == 'VFL_v2':
            plt.plot(x_axis, is_vfl_v2, '*-', label='VFLGAN')
        else:
            plt.plot(x_axis, is_real, '*-', label='real images')
    plt.legend()
    plt.savefig('IS.png', bbox_inches='tight')
