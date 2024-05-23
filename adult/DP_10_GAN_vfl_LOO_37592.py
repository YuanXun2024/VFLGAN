import argparse
import os
import numpy as np
import pandas as pd
from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader
from scipy.linalg import sqrtm

from torch.autograd import Variable
import torch.nn.functional as F

import torch.nn as nn
import torch.autograd as autograd
import torch
import random
import matplotlib.pyplot as plt
import multiprocessing as mp
import warnings 
warnings.filterwarnings('ignore')
torch.cuda.set_device('cuda:'+str(6))


def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=60, help="number of epochs of training")
parser.add_argument("--n_steps", type=int, default=671, help="number of steps of epoch")

parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0005, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=10, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--n_critic", type=int, default=5, help="number of training steps for discriminator per iter")
parser.add_argument("--clip_value", type=float, default=0.01, help="lower and upper clip value for disc. weights")
parser.add_argument("--sample_interval", type=int, default=400, help="interval betwen image samples")
parser.add_argument("--sigma", type=float, default=0.6128, help="interval betwen image samples")
opt = parser.parse_args()
SENSITIVITY = 2

cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
lambda_gp = 30


class CustomDataset(Dataset):
    def __init__(self, data, label, delet_target=None, transform=None):
        if delet_target is None:
            self.train_set = data
            self.train_labels = label
        else:
            self.train_set = np.delete(data, delet_target, axis=0)
            self.train_labels = np.delete(label, delet_target)
        self.transform = transform

    def __getitem__(self, index):
        img = np.array(self.train_set[index])
        if self.transform is not None:
            img = self.transform(img)
        return  torch.from_numpy(img)

    def __len__(self):
        return len(self.train_set)
    

def preprocess(df, delet_target=None):
    name_list = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
    if delet_target is not None:
        df.drop(index=delet_target, inplace=True)
    for head in name_list:
        df[head] = (df[head]-df[head].mean()) / df[head].std()
    return df.values


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


class Generator(nn.Module):
    def __init__(self, span_info):
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat, bias=False)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(opt.latent_dim, 32, normalize=False),
            *block(32, 64),
            *block(64, 128),
            nn.Linear(128, 53),
        )

        self.span_info = span_info

    def _apply_activate(self, data):
        data_t = []
        for span in self.span_info:
            st = span[0]
            ed = span[1]
            if ed-st == 1:
                transformed = data[:, st:ed]
            else:
                transformed = F.gumbel_softmax(data[:, st:ed], tau=1, hard=True)
            data_t.append(transformed)
        return torch.cat(data_t, dim=1)

    def forward(self, z):
        output = self.model(z)
        output = self._apply_activate(output)

        return output
    

class DiscriminatorClient(nn.Module):
    def __init__(self):
        super(DiscriminatorClient, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(53, 256, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 128, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, img):
        img_flat = img.view(img.shape[0], -1)
        latent = self.model(img_flat)
        return latent


class DiscriminatorPrivate(nn.Module):
    def __init__(self):
        super(DiscriminatorPrivate, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(128, 1, bias=False),
            # nn.LeakyReLU(0.2, inplace=True),
            # nn.Linear(64, 1),
        )

    def forward(self, latent):
        validity = self.model(latent)
        return validity


class DiscriminatorServer(nn.Module):
    def __init__(self):
        super(DiscriminatorServer, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(256, 128, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 1, bias=False),
        )

    def forward(self, latent):
        validity = self.model(latent)
        return validity



def compute_gradient_penalty_2(D_S, D_C1, D_C2, D_p1, D_p2, real_imgs_client_1, fake_imgs_client_1, real_imgs_client_2, fake_imgs_client_2):
    # Random weight term for interpolation between real and fake samples
    alpha = Tensor(np.random.random((real_imgs_client_1.size(0), 1)))
    # Get random interpolation between real and fake samples
    interpolates_1 = (alpha * real_imgs_client_1 + ((1 - alpha) * fake_imgs_client_1)).requires_grad_(True)
    interpolates_2 = (alpha * real_imgs_client_2 + ((1 - alpha) * fake_imgs_client_2)).requires_grad_(True)

    latent_1 = D_C1(interpolates_1)
    latent_2 = D_C2(interpolates_2)

    d_private_1 = D_p1(latent_1)
    d_private_2 = D_p2(latent_2)

    latent = torch.cat((latent_1, latent_2), dim=1)
    d_interpolates = D_S(latent)

    grad_C = Variable(Tensor(real_imgs_client_1.shape[0], 128).fill_(1.0), requires_grad=False)
    grad_S = Variable(Tensor(real_imgs_client_1.shape[0], 1).fill_(1.0), requires_grad=False)

    gradients_c1 = autograd.grad(
        outputs=latent_1,
        inputs=interpolates_1,
        grad_outputs=grad_C,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    gradients_c2 = autograd.grad(
        outputs=latent_2,
        inputs=interpolates_2,
        grad_outputs=grad_C,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    gradients_s = autograd.grad(
        outputs=d_interpolates,
        inputs=latent,
        grad_outputs=grad_S,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    gradients_cp1 = autograd.grad(
        outputs=d_private_1,
        inputs=latent_1,
        grad_outputs=grad_S,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    gradients_cp2 = autograd.grad(
        outputs=d_private_2,
        inputs=latent_2,
        grad_outputs=grad_S,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    gradients_c1 = gradients_c1.view(gradients_c1.size(0), -1)
    gradients_c2 = gradients_c2.view(gradients_c2.size(0), -1)
    gradients_cp1 = gradients_cp1.view(gradients_cp1.size(0), -1)
    gradients_cp2 = gradients_cp2.view(gradients_cp2.size(0), -1)
    gradients_s = gradients_s.view(gradients_s.size(0), -1)
    gradient_penalty = ((gradients_c1.norm(2, dim=1) - 1)**2).mean() + ((gradients_c2.norm(2, dim=1) - 1)**2).mean() \
                       + ((gradients_cp1.norm(2, dim=1) - 1)**2).mean() + ((gradients_cp2.norm(2, dim=1)-1)**2).mean() \
                       + ((gradients_s.norm(2, dim=1) - 1) ** 2).mean()
    # gradient_penalty = ((torch.cat([gradients_c1,gradients_cp1],1).norm(2, dim=1) - 1)**2).mean() + ((torch.cat([gradients_c2,gradients_cp2],1).norm(2, dim=1) - 1)**2).mean() \
    #                     + ((gradients_s.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


def train(model_list, optimizer_list, dataset, data_new, param_path="params/WGAN_centric_shadow/0/"):
    os.makedirs(param_path, exist_ok=True)

    fid_min = 1
    fid_list = []

    G_1 = model_list[0]
    G_2 = model_list[1]
    D_C1 = model_list[2]
    D_C2 = model_list[3]
    D_p1 = model_list[4]
    D_p2 = model_list[5]
    D_S = model_list[6]

    optimizer_G1 = optimizer_list[0]
    optimizer_G2 = optimizer_list[1]
    optimizer_D1 = optimizer_list[2]
    optimizer_D2 = optimizer_list[3]
    optimizer_Dp1 = optimizer_list[4]
    optimizer_Dp2 = optimizer_list[5]
    optimizer_DS = optimizer_list[6]

    ### clip
    ### add noise    

    def func(grad):
        clip_norm = 1.5 / opt.batch_size
        grad_input_norm = torch.norm(grad, p=2, keepdim=True)
        clip_coef = clip_norm / (grad_input_norm + 1e-10)
        clip_coef = torch.min(clip_coef, torch.ones_like(clip_coef))
        grad = grad*clip_coef + clip_norm * SENSITIVITY * opt.sigma * torch.randn(
            grad.shape).cuda().float()
        return grad

    param_D1 = D_C1.parameters().__next__()

    param_D1.register_hook(
       func
    )

    param_D2 = D_C2.parameters().__next__()
    param_D2.register_hook(
        func
    )

    # Training
    batches_done = 0
    for epoch in range(opt.n_epochs):
        for i in range(opt.n_steps):
            index = random.sample(range(dataset.__len__()), opt.batch_size)
            imgs = dataset.__getitem__(index=index)
            G_1.train()
            G_2.train()
            D_C1.train()
            D_C2.train()
            D_p1.train()
            D_p2.train()
            D_S.train()
            # Configure input
            imgs_client_1 = imgs[:, :53]
            imgs_client_2 = imgs[:, 53:]
            real_imgs_client_1 = Variable(imgs_client_1.type(Tensor))
            real_imgs_client_2 = Variable(imgs_client_2.type(Tensor))

            optimizer_D1.zero_grad()
            optimizer_D2.zero_grad()
            optimizer_Dp1.zero_grad()
            optimizer_Dp2.zero_grad()
            optimizer_DS.zero_grad()

            # Sample noise as generator input
            z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))

            # Generate a batch of images
            fake_imgs_client_1 = G_1(z)
            fake_imgs_client_2 = G_2(z)

            # Real images
            latent_1 = D_C1(real_imgs_client_1)
            latent_2 = D_C2(real_imgs_client_2)

            real_private_validity_1 = D_p1(latent_1)
            real_private_validity_2 = D_p2(latent_2)

            latent_real = torch.cat((latent_1, latent_2), dim=1)

            real_validity = D_S(latent_real)
            # Fake images
            latent_f1 = D_C1(fake_imgs_client_1)
            latent_f2 = D_C2(fake_imgs_client_2)

            latent_fake = torch.cat((latent_f1, latent_f2), dim=1)

            fake_private_validity_1 = D_p1(latent_f1)
            fake_private_validity_2 = D_p2(latent_f2)

            fake_validity = D_S(latent_fake)

            gradient_penalty_DS = compute_gradient_penalty_2(D_S, D_C1, D_C2, D_p1, D_p2, real_imgs_client_1.data,
                                                             fake_imgs_client_1.data, real_imgs_client_2.data,
                                                             fake_imgs_client_2.data)

            # Adversarial loss
            lambda_dp = 1.
            d_loss = -(1)*torch.mean(real_validity) - lambda_dp*torch.mean(real_private_validity_1) - lambda_dp*torch.mean(real_private_validity_2) + \
                      (1)*torch.mean(fake_validity) + lambda_dp*torch.mean(fake_private_validity_1) + lambda_dp*torch.mean(fake_private_validity_2) + \
                      lambda_gp * gradient_penalty_DS

            d_loss.backward()
            optimizer_D1.step()
            optimizer_D2.step()
            optimizer_Dp1.step()
            optimizer_Dp2.step()
            optimizer_DS.step()

            optimizer_G1.zero_grad()
            optimizer_G2.zero_grad()

            # Train the generator every n_critic steps
            if i % opt.n_critic == 0:
                # Generate a batch of images
                fake_imgs_client_1 = G_1(z)
                fake_imgs_client_2 = G_2(z)

                # Train on fake images
                latent_1 = D_C1(fake_imgs_client_1)
                latent_2 = D_C2(fake_imgs_client_2)
                latent_fake = torch.cat((latent_1, latent_2), dim=1)

                fake_private_validity_1 = D_p1(latent_1)
                fake_private_validity_2 = D_p2(latent_2)

                fake_validity = D_S(latent_fake)

                g_loss = -(1)*torch.mean(fake_validity) - lambda_dp * (torch.mean(fake_private_validity_1) + torch.mean(fake_private_validity_2))
                g_loss.backward()

                optimizer_G1.step()
                optimizer_G2.step()

        G_1.eval()
        G_2.eval()

        z = Variable(Tensor(np.random.normal(0, 1, (data_new.shape[0], opt.latent_dim))))

        fake_imgs_client_1 = G_1(z)
        fake_imgs_client_2 = G_2(z)
        fake_data = torch.cat((fake_imgs_client_1, fake_imgs_client_2), dim=1)

        fake_data = fake_data.cpu().detach().numpy()

        fid = calculate_fid(data_new, fake_data)
        fid_list += [fid]

        if fid < fid_min:
            torch.save(G_1.state_dict(), param_path + "G_1.pth")
            torch.save(G_2.state_dict(), param_path + "G_2.pth")
            torch.save(D_C1.state_dict(), param_path + "D_C1.pth")
            torch.save(D_C2.state_dict(), param_path + "D_C2.pth")
            fid_min = fid
            min_epoch = epoch

    title = 'min_fid_' + str(min(fid_list)) + '_epoch_' + str(min_epoch) + '.npy'
    fid_np = np.array(fid_list)
    np.save(param_path+title, fid_np)

    t = [i for i in range(len(fid_list))]
    plt.plot(t, fid_list)
    plt.title(title)
    plt.xlabel('epoch')
    plt.ylabel('fid score')
    plt.savefig(param_path + 'fid.jpg')
    return fid_min


def initialization(seed, delet_target=None):
    set_random_seed(seed)

    df = pd.read_csv('data/adult.csv')
    data = preprocess(df, delet_target=delet_target)
    labels = np.zeros(data.shape[0])

    data_new = np.zeros_like(data)

    # data stored in client 1
    data_new[:, 0] = data[:, 0]
    data_new[:, 1:1 + 7] = data[:, 6:6 + 7]
    data_new[:, 8] = data[:, 1]
    data_new[:, 9:9 + 16] = data[:, 13:13 + 16]
    data_new[:, 25] = data[:, 2]
    data_new[:, 26:26 + 7] = data[:, 29:29 + 7]
    data_new[:, 33:33 + 14] = data[:, 36:36 + 14]
    data_new[:, 47:47 + 6] = data[:, 50:50 + 6]

    # data stored in client 2
    data_new[:, 53:53 + 5] = data[:, 56:56 + 5]
    data_new[:, 58:58 + 2] = data[:, 61:61 + 2]
    data_new[:, 60] = data[:, 3]
    data_new[:, 61] = data[:, 4]
    data_new[:, 62] = data[:, 5]
    data_new[:, 63:63 + 41] = data[:, 63:63 + 41]
    data_new[:, 104:104 + 2] = data[:, 104:104 + 2]

    dataset = CustomDataset(data_new, labels)
    batch_size = opt.batch_size

    # Initialize generator and discriminator
    span_info_1 = [(0, 0 + 1), (1, 1 + 7), (8, 8 + 1), (9, 9 + 16), (25, 25 + 1), (26, 26 + 7), (33, 33 + 14),
                   (47, 47 + 6)]
    span_info_2 = [(0, 0 + 5), (58 - 53, 58 + 2 - 53), (60 - 53, 60 + 1 - 53), (61 - 53, 61 + 1 - 53),
                   (62 - 53, 62 + 1 - 53), (63 - 53, 63 + 41 - 53), (104 - 53, 104 + 2 - 53)]

    G_1 = Generator(span_info_1)
    G_2 = Generator(span_info_2)

    D_C1 = DiscriminatorClient()
    D_C2 = DiscriminatorClient()

    D_p1 = DiscriminatorPrivate()
    D_p2 = DiscriminatorPrivate()

    D_S = DiscriminatorServer()

    if cuda:
        G_1.cuda()
        G_2.cuda()
        D_C1.cuda()
        D_C2.cuda()
        D_p1.cuda()
        D_p2.cuda()
        D_S.cuda()

    optimizer_G1 = torch.optim.Adam(G_1.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_G2 = torch.optim.Adam(G_2.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D1 = torch.optim.Adam(D_C1.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D2 = torch.optim.Adam(D_C2.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_Dp1 = torch.optim.Adam(D_p1.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_Dp2 = torch.optim.Adam(D_p2.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_DS = torch.optim.Adam(D_S.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    model_list = [G_1, G_2, D_C1, D_C2, D_p1, D_p2, D_S]
    optimizer_list = [optimizer_G1, optimizer_G2, optimizer_D1, optimizer_D2, optimizer_Dp1, optimizer_Dp2, optimizer_DS]

    return model_list, optimizer_list, dataset, data_new


# def main(model_list, optimizer_list, dataloader, data_new):
#     for i in tqdm(range(10)):
#         seed = 10*i+index
#         param_path = "params/WGAN_vfl_LOO_33914/" + str(seed) + '/'
#         train(model_list, optimizer_list, dataloader, data_new, param_path=param_path)


if __name__ == '__main__':
    target = 37592
    print('DP-10-LOO:', target)

    mp = mp.get_context('spawn')

    for i in tqdm(range(25)):
        for j in range(4):
            seed = 4*i+j
            model_list, optimizer_list, dataset, data_new = initialization(seed, delet_target=target)
            if target is not None:
                param_path = 'params/DP_WGAN_vfl_LOO_epsilon_10_' + str(target) + '/' + str(seed) + '/'
            else:
                param_path = 'params/DP_WGAN_vfl_shadow_epsilon_10' + '/' + str(seed) + '/'
            if j == 0:
                p0 = mp.Process(target=train, args=(model_list,optimizer_list,dataset,data_new,param_path))
            elif j == 1:
                p1 = mp.Process(target=train, args=(model_list, optimizer_list, dataset, data_new, param_path))
            elif j == 2:
                p2 = mp.Process(target=train, args=(model_list, optimizer_list, dataset, data_new, param_path))
            elif j == 3:
                p3 = mp.Process(target=train, args=(model_list, optimizer_list, dataset, data_new, param_path))
           
        p0.start()
        p1.start()
        p2.start()
        p3.start()
        p0.join()
        p1.join()
        p2.join()
        p3.join()
