import argparse
import os
import numpy as np
import math
import sys
import copy

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch
import random
random.seed(0)

torch.manual_seed(0)
np.random.seed(0)

os.makedirs("images", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=300, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")  # 0.0002
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--n_critic", type=int, default=5, help="number of training steps for discriminator per iter")
parser.add_argument("--clip_value", type=float, default=0.01, help="lower and upper clip value for disc. weights")
parser.add_argument("--sample_interval", type=int, default=400, help="interval betwen image samples")
parser.add_argument("--aggregation_interval", type=int, default=10, help="interval betwen image samples")
parser.add_argument("--train", action="store_true")

opt = parser.parse_args()
print(opt)

img_shape = (opt.channels, opt.img_size // 2, opt.img_size)

cuda = True if torch.cuda.is_available() else False


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(opt.latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.shape[0], *img_shape)
        return img


class DiscriminatorClient(nn.Module):
    def __init__(self):
        super(DiscriminatorClient, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
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
            nn.Linear(256, 1)
        )

    def forward(self, latent):
        validity = self.model(latent)
        return validity


class DiscriminatorServer(nn.Module):
    def __init__(self):
        super(DiscriminatorServer, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
        )

    def forward(self, latent):
        validity = self.model(latent)
        return validity


# Loss weight for gradient penalty
lambda_gp = 40

# Configure data loader
os.makedirs("./data/mnist", exist_ok=True)
os.makedirs("./generated_data/training", exist_ok=True)
dataloader = torch.utils.data.DataLoader(
    datasets.MNIST(
        "./data/mnist",
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.Resize(opt.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        ),
    ),
    batch_size=opt.batch_size,
    shuffle=True,
)


Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


def compute_gradient_penalty(D, real_samples, fake_samples):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = Tensor(np.random.random((real_samples.size(0), 1)))
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = Variable(Tensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False)
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


def compute_gradient_penalty_2(D_S, D_C1, D_C2, D_p1, D_p2, real_imgs_client_1, fake_imgs_client_1, real_imgs_client_2, fake_imgs_client_2):
    # Random weight term for interpolation between real and fake samples
    alpha = Tensor(np.random.random((real_imgs_client_1.size(0), 1, 1, 1)))
    # Get random interpolation between real and fake samples
    interpolates_1 = (alpha * real_imgs_client_1 + ((1 - alpha) * fake_imgs_client_1)).requires_grad_(True)
    interpolates_2 = (alpha * real_imgs_client_2 + ((1 - alpha) * fake_imgs_client_2)).requires_grad_(True)

    latent_1 = D_C1(interpolates_1)
    latent_2 = D_C2(interpolates_2)

    d_private_1 = D_p1(latent_1)
    d_private_2 = D_p2(latent_2)

    latent = torch.cat((latent_1, latent_2), dim=1)
    d_interpolates = D_S(latent)

    grad_C = Variable(Tensor(real_imgs_client_1.shape[0], 256).fill_(1.0), requires_grad=False)
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
    gradient_penalty = ((gradients_c1.norm(2, dim=1) - 1) ** 2).mean() + ((gradients_c2.norm(2, dim=1) - 1) ** 2).mean() \
                       + ((gradients_cp1.norm(2, dim=1) - 1) ** 2).mean() + ((gradients_cp2.norm(2, dim=1) - 1) ** 2).mean() + ((gradients_s.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


def main():
    # Initialize generator and discriminator
    G_1 = Generator()
    G_2 = Generator()

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

    # Optimizers
    optimizer_G1 = torch.optim.Adam(G_1.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_G2 = torch.optim.Adam(G_2.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D1 = torch.optim.Adam(D_C1.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D2 = torch.optim.Adam(D_C2.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_Dp1 = torch.optim.Adam(D_p1.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_Dp2 = torch.optim.Adam(D_p2.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_DS = torch.optim.Adam(D_S.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    # Training
    batches_done = 0
    for epoch in range(opt.n_epochs):

        for i, (imgs, _) in enumerate(dataloader):

            # Configure input

            imgs_client_1 = imgs[:, :, :opt.img_size//2, :]
            imgs_client_2 = imgs[:, :, opt.img_size // 2:, :]
            real_imgs_client_1 = Variable(imgs_client_1.type(Tensor))
            real_imgs_client_2 = Variable(imgs_client_2.type(Tensor))

            # ---------------------
            #  Train Discriminator
            # ---------------------

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
            latent_1 = D_C1(fake_imgs_client_1)
            latent_2 = D_C2(fake_imgs_client_2)

            latent_fake = torch.cat((latent_1, latent_2), dim=1)

            fake_private_validity_1 = D_p1(latent_1)
            fake_private_validity_2 = D_p2(latent_2)

            fake_validity = D_S(latent_fake)

            # Gradient penalty
            # gradient_penalty_D1 = compute_gradient_penalty(D_C1, real_imgs_client_1.data, fake_imgs_client_1.data)
            # gradient_penalty_D2 = compute_gradient_penalty(D_C2, real_imgs_client_2.data, fake_imgs_client_2.data)
            gradient_penalty_DS = compute_gradient_penalty_2(D_S, D_C1, D_C2, D_p1, D_p2, real_imgs_client_1.data,
                                                             fake_imgs_client_1.data, real_imgs_client_2.data,
                                                             fake_imgs_client_2.data)

            # Adversarial loss
            lambda_dp = 1.0
            d_loss = -torch.mean(real_validity) - lambda_dp*torch.mean(real_private_validity_1) - lambda_dp*torch.mean(real_private_validity_2) + \
                      torch.mean(fake_validity) + lambda_dp*torch.mean(fake_private_validity_1) + lambda_dp*torch.mean(fake_private_validity_2) + \
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

                # -----------------
                #  Train Generator
                # -----------------

                # Generate a batch of images
                fake_imgs_client_1 = G_1(z)
                fake_imgs_client_2 = G_2(z)

                # Loss measures generator's ability to fool the discriminator
                # Train on fake images
                latent_1 = D_C1(fake_imgs_client_1)
                latent_2 = D_C2(fake_imgs_client_2)
                latent_fake = torch.cat((latent_1, latent_2), dim=1)

                fake_private_validity_1 = D_p1(latent_1)
                fake_private_validity_2 = D_p2(latent_2)

                fake_validity = D_S(latent_fake)

                g_loss = -torch.mean(fake_validity) - lambda_dp * (torch.mean(fake_private_validity_1) + torch.mean(fake_private_validity_2))
                g_loss.backward()

                optimizer_G1.step()
                optimizer_G2.step()

                print("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]" % (epoch, opt.n_epochs, i, len(dataloader)
                                                                                 , d_loss.item(), g_loss.item()))

                if batches_done % opt.sample_interval == 0:
                    fake_imgs_client_1 = G_1(z)
                    fake_imgs_client_2 = G_2(z)

                    fake_imgs = torch.cat((fake_imgs_client_1, fake_imgs_client_2), dim=2)
                    save_image(fake_imgs.data[:25], "generated_data/training/%d.png" % batches_done, nrow=5, normalize=True)

                batches_done += opt.n_critic

        if (epoch + 1) % 10 == 0:
            torch.save(G_1.state_dict(), "params/WGAN_GP_real_VFL_v2_40/G_1_%d.pth" % (epoch + 1))
            torch.save(G_2.state_dict(), "params/WGAN_GP_real_VFL_v2_40/G_2_%d.pth" % (epoch + 1))
            # torch.save(D_C1.state_dict(), "params/WGAN_GP_real_VFL_v2_1.0_60/D_C1_%d.pth" % (epoch + 1))
            # torch.save(D_C2.state_dict(), "params/WGAN_GP_real_VFL_v2_1.0_60/D_C2_%d.pth" % (epoch + 1))
            # torch.save(D_p1.state_dict(), "params/WGAN_GP_real_VFL_v2_1.0_60/D_p1_%d.pth" % (epoch + 1))
            # torch.save(D_p2.state_dict(), "params/WGAN_GP_real_VFL_v2_1.0_60/D_p2_%d.pth" % (epoch + 1))
            # torch.save(D_S.state_dict(), "params/WGAN_GP_real_VFL_v2_1.0_60/D_S_%d.pth" % (epoch + 1))


def inference(epoch=56):
    G_1 = Generator().cuda()
    G_2 = Generator().cuda()

    name_1 = 'G_1_' + str(epoch*5) + '.pth'
    name_2 = 'G_2_' + str(epoch*5) + '.pth'
    G_1.load_state_dict(torch.load("params/WGAN_GP_real_VFL_v2_40/" + name_1))
    G_2.load_state_dict(torch.load("params/WGAN_GP_real_VFL_v2_40/" + name_2))
    G_1.eval()
    G_2.eval()

    z = Variable(Tensor(np.random.normal(0, 1, (64, opt.latent_dim))))

    fake_imgs_client_1 = G_1(z)
    fake_imgs_client_2 = G_2(z)

    fake_imgs = torch.cat((fake_imgs_client_1, fake_imgs_client_2), dim=2)
    save_image(fake_imgs.data[:25], "generated_data/VFLGAN.png", nrow=5, normalize=True)


if __name__ == "__main__":
    if opt.train:
        main()
    else:
        inference()
