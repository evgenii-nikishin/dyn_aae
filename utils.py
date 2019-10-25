import os
import joblib
import numpy as np
from itertools import zip_longest

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

def save_model(model, path, domain):
    os.makedirs(os.path.join(path, domain), exist_ok=True)
    for name, network in model.items():
        torch.save(network, os.path.join(path, domain, '{}.pkl'.format(name)))

def load_model(path, domain):
    model = {}
    for name in os.listdir(os.path.join(path, domain)):
        model[name[:-4]] = torch.load(os.path.join(path, domain, name))
    return model

def save_metrics(metrics_dict, path, domain):
    os.makedirs(os.path.join(path, domain), exist_ok=True)
    joblib.dump(metrics_dict, os.path.join(path, domain, 'metrics_dict.pkl'))

def load_metrics(path, domain):
    return joblib.load(os.path.join(path, domain, 'metrics_dict.pkl'))

def reparameterization(mu, logvar):
    std = torch.exp(logvar / 2)
    sampled_z = Tensor(np.random.normal(0, 1, (mu.size(0), mu.size(1))))
    z = sampled_z * std + mu
    return z


def grouper(iterable, n, fillvalue=None):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)

def z_separation_accuracy(encoder, obs, encoder_2=None, obs_2=None):
    z_s = encoder(obs)
    if encoder_2 is None:
        z = Tensor(np.random.normal(0, 1, (obs.shape[0], encoder.latent_dim)))
    else:
        z = encoder_2(obs_2)
    
    X = torch.cat([z_s, z], dim=0).data.cpu().numpy()
    y = np.concatenate([np.ones(z_s.shape[0]), np.zeros(z.shape[0])])
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    clf_rf = RandomForestClassifier()
    clf_rf.fit(X_train, y_train)
    preds_rf = clf_rf.predict(X_test)
    rf_acc = np.mean(preds_rf == y_test)
    print('rf acc:', rf_acc)
    
    clf_lr = LogisticRegression()
    clf_lr.fit(X_train, y_train)
    preds_lr = clf_lr.predict(X_test)
    lr_acc = np.mean(preds_lr == y_test)
    print('lr acc:', lr_acc)

    return rf_acc, lr_acc

def sample_image(decoder, n_row, batches_done):
    """Saves a grid of generated digits"""
    # Sample noise
    os.makedirs("images", exist_ok=True)
    z = Tensor(np.random.normal(0, 1, (n_row ** 2, decoder.latent_dim)))
    gen_imgs = decoder(z)
    save_image(gen_imgs.data, "images/%d.png" % batches_done, nrow=n_row, normalize=True)