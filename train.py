import argparse
import os
import numpy as np
import math
import joblib
import itertools
import matplotlib.pyplot as plt

import torch.nn as nn
import torch.nn.functional as F
import torch

from models import Encoder, Decoder, Dynamics, Policy, Discriminator
from utils import Tensor, reparameterization, grouper, z_separation_accuracy, sample_image
from utils import save_model, load_model, save_metrics, load_metrics
from gen_data import Data

def init_training(opt):
    # Initialize losses
    losses = {
        'adversarial': torch.nn.BCELoss(),
        'pixelwise': torch.nn.L1Loss(),
        'action': torch.nn.NLLLoss()
    }

    img_shape = (1, opt.img_size, opt.img_size)
    # Initialize models
    encoder = Encoder(img_shape, opt.latent_dim)
    decoder = Decoder(img_shape, opt.latent_dim)
    discriminator = Discriminator(opt.latent_dim)
    model = {
        'enc': encoder,
        'dec': decoder,
        'discr': discriminator
    }
    
    if opt.domain == 'source':
        pol = Policy(opt.latent_dim, ac_size=3)
        model['pol'] = pol
    
    if opt.use_dynamics:
        decoder_next = Decoder(img_shape, opt.latent_dim)
        model['dec_next'] = decoder_next
        if opt.domain == 'source':
            dyn = Dynamics(opt.latent_dim, ac_size=3, ac_embed_size=10)
            model['dyn'] = dyn
    
    # move to GPU
    if opt.cuda:
        for loss in losses.values():
            loss.cuda()
        for network in model.values():
            network.cuda()
                
    # Optimizers
    G_params = []
    for name, network in model.items():
        G_params += [network.parameters()] if name != 'discr' else []
    G_params = itertools.chain(*G_params)
    
    optimizer_G = torch.optim.Adam(G_params, lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D = torch.optim.Adam(model['discr'].parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    
    # metrics
    metrics_dict = {
        'adv_losses': [],
        'pix_losses': [],
        'ac_losses': [],
        'g_losses': [],
        'd_losses': [],
        'rf_z_sep_accs': [],
        'pol_accs': []
    }
    if opt.use_dynamics:
        metrics_dict['pix_next_losses'] = []
        
    return model, losses, optimizer_G, optimizer_D, metrics_dict


def train(opt, source_model=None):
    opt.use_dynamics = not opt.no_dynamics
    train_data_s, train_data_t, test_data_s, test_data_t = joblib.load(os.path.join(opt.data_path, 'all_data.pkl'))
    model, losses, optimizer_G, optimizer_D, metrics_dict = init_training(opt)
    if opt.domain == 'source':
        source_model = model

    train_data = train_data_s if opt.domain == 'source' else train_data_t
    test_data = test_data_s if opt.domain == 'source' else test_data_t
    for epoch in range(opt.n_epochs):
        gen = grouper(np.random.permutation(len(train_data.obs)), opt.batch_size)
        num_batches = int(np.ceil(len(train_data.obs)/opt.batch_size))

        for batch_idx, data_idxs in enumerate(gen):
            data_idxs = list(filter(None, data_idxs))
            obs = train_data.obs[data_idxs]
            acs = train_data.acs[data_idxs]
            next_obs = train_data.obs_[data_idxs]
            if opt.cuda:
                obs, acs, next_obs = obs.cuda(), acs.cuda(), next_obs.cuda()

            # Adversarial ground truths
            valid = Tensor(obs.shape[0], 1).fill_(1.0).detach()
            fake = Tensor(obs.shape[0], 1).fill_(0.0).detach()

            # Configure input
            real_imgs = obs.type(Tensor)
            real_next_imgs = next_obs.type(Tensor)

            # -----------------
            #  Train Generator
            # -----------------

            optimizer_G.zero_grad()

            encoded_imgs = model['enc'](real_imgs)
            if opt.domain == 'source':
                # detached, i.e. policy training does not affect encoder
                pred_acs = source_model['pol'](encoded_imgs.detach())
            decoded_imgs = model['dec'](encoded_imgs)
            if opt.use_dynamics:
                encoded_next_imgs = source_model['dyn'](encoded_imgs, acs)
                decoded_next_imgs = model['dec_next'](encoded_next_imgs)
            if opt.domain == 'target':
                with torch.no_grad():
                    pred_acs = source_model['pol'](encoded_imgs.detach())
                    ac_loss = losses['action'](pred_acs, acs)

            # Loss measures generator's ability to fool the discriminator
            adv_loss = losses['adversarial'](model['discr'](encoded_imgs), valid)
            pix_loss = losses['pixelwise'](decoded_imgs, real_imgs)
            
            if opt.domain == 'source':
                ac_loss = losses['action'](pred_acs, acs)
            if opt.use_dynamics:
                pix_next_loss = losses['pixelwise'](decoded_next_imgs, real_next_imgs)
                g_loss = opt.adv_coef * adv_loss + (1-opt.adv_coef)/2 * pix_loss + (1-opt.adv_coef)/2 * pix_next_loss
            else:
                g_loss = opt.adv_coef * adv_loss + (1-opt.adv_coef) * pix_loss
            
            if opt.domain == 'source':
                g_loss = 0.5*g_loss + 0.5*ac_loss
            
            g_loss.backward()
            optimizer_G.step()
            
            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()
            
            # Sample noise as discriminator ground truth
            if opt.domain == 'source':
                z = Tensor(np.random.normal(0, 1, (obs.shape[0], opt.latent_dim)))
            elif opt.domain == 'target':
                obs_s = train_data_s.obs[data_idxs].cuda() if opt.cuda else train_data_s.obs[data_idxs]
                z = source_model['enc'](obs_s.type(Tensor))


            # Measure discriminator's ability to classify real from generated samples
            real_loss = losses['adversarial'](model['discr'](z), valid)
            fake_loss = losses['adversarial'](model['discr'](encoded_imgs.detach()), fake)
            d_loss = 0.5 * (real_loss + fake_loss)

            d_loss.backward()
            optimizer_D.step()
            
            if batch_idx % opt.log_interval == 0:
                print(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [adv loss: %f] [pix loss: %f] [ac loss: %f] [G loss: %f]"
                    % (epoch, opt.n_epochs, batch_idx, num_batches, d_loss.item(), adv_loss.item(), pix_loss.item(), ac_loss.item(), g_loss.item())
                )

            batches_done = epoch * num_batches + batch_idx
            if batches_done % opt.sample_interval == 0:
                sample_image(model['dec'], n_row=10, batches_done=batches_done)

            metrics_dict['g_losses'].append(g_loss.item())
            metrics_dict['pix_losses'].append(pix_loss.item())
            metrics_dict['adv_losses'].append(adv_loss.item())
            metrics_dict['d_losses'].append(d_loss.item())
            metrics_dict['ac_losses'].append(ac_loss.item())
            if opt.use_dynamics:
                metrics_dict['pix_next_losses'].append(pix_next_loss.item())

        with torch.no_grad():
            # careful, all test data may be too large for a gpu
            test_obs = test_data.obs.cuda() if opt.cuda else test_data.obs
            if opt.domain == 'source':
                rf_acc, _ = z_separation_accuracy(model['enc'], test_obs)                
            elif opt.domain == 'target':
                test_obs_s = test_data_s.obs.cuda() if opt.cuda else test_data_s.obs
                rf_acc, _ = z_separation_accuracy(model['enc'], test_obs, source_model['enc'], test_obs_s)
            pred_acs = source_model['pol'](model['enc'](test_obs.type(Tensor)))
            metrics_dict['rf_z_sep_accs'].append(rf_acc)
            pol_acc = (torch.max(pred_acs.cpu(), 1)[1] == test_data.acs).float().mean().item()
            metrics_dict['pol_accs'].append(pol_acc)
            
    return model, metrics_dict

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=100, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
    parser.add_argument("--latent_dim", type=int, default=10, help="dimensionality of the latent code")
    parser.add_argument("--domain", default='source', choices=['source', 'target'], help="domain to train")
    parser.add_argument('--no_dynamics', action='store_true', help='dynamics usage flag')
    parser.add_argument('--adv_coef', type=float, default=0.01, help='adversarial posterior matching coef')
    parser.add_argument('--log_interval', type=int, default=100, help='how frequently output statistics')
    parser.add_argument("--data_path", default='./data/mnist_prep', help="path to processed data")
    parser.add_argument("--model_path", default='./models', help="save a trained model to this path")
    parser.add_argument('--log_path', default='./logs', help='path to logs')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument("--sample_interval", type=int, default=400, help="interval between image sampling")
    opt = parser.parse_args()
    
    train_data_s, train_data_t, test_data_s, test_data_t = joblib.load(os.path.join(opt.data_path, 'all_data.pkl'))
    
    opt.cuda = True if torch.cuda.is_available() else False
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    
    if opt.domain == 'source':
        model, metrics_dict = train(opt)
    elif opt.domain == 'target':
        model_s = load_model(opt.model_path, 'source')
        model, metrics_dict = train(opt, model_s)
        
    save_model(model, opt.model_path, opt.domain)
    save_metrics(metrics_dict, opt.log_path, opt.domain)
