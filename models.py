import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import reparameterization


class Encoder(nn.Module):
    def __init__(self, img_shape, latent_dim):
        super(Encoder, self).__init__()
        self.img_shape = img_shape
        self.latent_dim = latent_dim

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(self.img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.mu = nn.Linear(512, self.latent_dim)
        self.logvar = nn.Linear(512, self.latent_dim)

    def forward(self, img):
        img_flat = img.view(img.shape[0], -1)
        x = self.model(img_flat)
        mu = self.mu(x)
        logvar = self.logvar(x)
        assert mu.size(1) == self.latent_dim
        z = reparameterization(mu, logvar)
        return z


class Decoder(nn.Module):
    def __init__(self, img_shape, latent_dim):
        super(Decoder, self).__init__()
        self.img_shape = img_shape
        self.latent_dim = latent_dim

        self.model = nn.Sequential(
            nn.Linear(self.latent_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, int(np.prod(self.img_shape))),
            nn.Tanh(),
        )

    def forward(self, z):
        img_flat = self.model(z)
        img = img_flat.view(img_flat.shape[0], *self.img_shape)
        return img


class Discriminator(nn.Module):
    def __init__(self, latent_dim):
        super(Discriminator, self).__init__()
        self.latent_dim = latent_dim

        self.model = nn.Sequential(
            nn.Linear(self.latent_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        return self.model(z)


class Dynamics(nn.Module):
    def __init__(self, latent_dim, ac_size, ac_embed_size):
        super(Dynamics, self).__init__()
        self.latent_dim = latent_dim
        self.ac_size = ac_size
        self.ac_embed_size = ac_embed_size
        self.ac_embed = nn.Embedding(ac_size, ac_embed_size)

        self.model = nn.Sequential(
            nn.Linear(self.latent_dim + self.ac_embed_size, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, self.latent_dim),
        )

    def forward(self, z, a):
        return self.model(torch.cat([z, self.ac_embed(a)], dim=-1))
    

class Policy(nn.Module):
    def __init__(self, latent_dim, ac_size):
        super(Policy, self).__init__()
        self.latent_dim = latent_dim
        self.ac_size = ac_size

        self.model = nn.Sequential(
            nn.Linear(self.latent_dim, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, self.ac_size),
        )

    def forward(self, z):
        return F.log_softmax(self.model(z), -1)