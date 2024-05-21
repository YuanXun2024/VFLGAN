"""VFLGAN_MNIST module."""
# third party
import argparse
import os
import torch
import numpy as np
import pandas as pd
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
import random


img_shape = (1, 14, 28)

class Generator(nn.Module):
    def __init__(self, latent_dim):
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(latent_dim, 128, normalize=False),
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


class VFLGANMNIST():
    def __init__(
        self,
        latent_dim: int,
        cuda: bool = True,
    ) -> None:
        super().__init__()

        if not cuda or not torch.cuda.is_available():
            device = "cpu"
            self.Tensor = torch.FloatTensor
        else:
            device = "cuda"
            self.Tensor = torch.cuda.FloatTensor

        self._device = torch.device(device)
        self.latent_dim = latent_dim
        

    @staticmethod
    def _compute_gradient_penalty_2(self, D_S, D_C1, D_C2, D_p1, D_p2, real_imgs_client_1, fake_imgs_client_1, real_imgs_client_2, fake_imgs_client_2):
        # Random weight term for interpolation between real and fake samples
        alpha = self.Tensor(np.random.random((real_imgs_client_1.size(0), 1, 1, 1)))
        # Get random interpolation between real and fake samples
        interpolates_1 = (alpha * real_imgs_client_1 + ((1 - alpha) * fake_imgs_client_1)).requires_grad_(True)
        interpolates_2 = (alpha * real_imgs_client_2 + ((1 - alpha) * fake_imgs_client_2)).requires_grad_(True)

        latent_1 = D_C1(interpolates_1)
        latent_2 = D_C2(interpolates_2)

        d_private_1 = D_p1(latent_1)
        d_private_2 = D_p2(latent_2)

        latent = torch.cat((latent_1, latent_2), dim=1)
        d_interpolates = D_S(latent)

        grad_C = Variable(self.Tensor(real_imgs_client_1.shape[0], 256).fill_(1.0), requires_grad=False)
        grad_S = Variable(self.Tensor(real_imgs_client_1.shape[0], 1).fill_(1.0), requires_grad=False)

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

    def fit(self, train_data, lr, b1, b2):
        """Fit the CTGAN Synthesizer models to the training data.

        Args:
            train_data (numpy.ndarray or pandas.DataFrame):
                Training Data. It must be a 2-dimensional numpy array or a pandas.DataFrame.
        """
        self.G_1 = Generator(self.latent_dim).to(self._device)
        self.G_2 = Generator(self.latent_dim).to(self._device)

        self.D_C1 = DiscriminatorClient().to(self._device)
        self.D_C2 = DiscriminatorClient().to(self._device)

        self.D_p1 = DiscriminatorPrivate().to(self._device)
        self.D_p2 = DiscriminatorPrivate().to(self._device)

        self.D_S = DiscriminatorServer().to(self._device)

        optimizer_G1 = torch.optim.Adam(self.G_1.parameters(), lr=lr, betas=(b1, b2))
        optimizer_G2 = torch.optim.Adam(self.G_2.parameters(), lr=lr, betas=(b1, b2))
        optimizer_D1 = torch.optim.Adam(self.D_C1.parameters(), lr=lr, betas=(b1, b2))
        optimizer_D2 = torch.optim.Adam(self.D_C2.parameters(), lr=lr, betas=(b1, b2))
        optimizer_Dp1 = torch.optim.Adam(self.D_p1.parameters(), lr=lr, betas=(b1, b2))
        optimizer_Dp2 = torch.optim.Adam(self.D_p2.parameters(), lr=lr, betas=(b1, b2))
        optimizer_DS = torch.optim.Adam(self.D_S.parameters(), lr=lr, betas=(b1, b2))
        
        return self

    def generate(self, n: int) -> pd.DataFrame:
        """Sample data similar to the training data.

        Choosing a condition_column and condition_value will increase the probability of the
        discrete condition_value happening in the condition_column.

        Args:
            count (int):
                Number of rows to sample.
        Returns:
            numpy.ndarray or pandas.DataFrame
        """
        if (
            self._data_sampler is None
            or self._generator is None
            or self._transformer is None
        ):
            raise RuntimeError("Train the model first")

        steps = n // self._batch_size + 1
        data = []
        for i in range(steps):
            mean = torch.zeros(self._batch_size, self._embedding_dim)
            std = mean + 1
            fakez = torch.normal(mean=mean, std=std).to(self._device)

            condvec = self._data_sampler.sample_original_condvec(self._batch_size)

            if condvec is None:
                pass
            else:
                c1 = condvec
                c1 = torch.from_numpy(c1).to(self._device)
                fakez = torch.cat([fakez, c1], dim=1)

            fake = self._generator(fakez)
            fakeact = self._apply_activate(fake)
            data.append(fakeact.detach().cpu().numpy())

        data = np.concatenate(data, axis=0)
        data = data[:n]

        return self._transformer.inverse_transform(data)

