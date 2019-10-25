import os
import joblib
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader


class Data:
    def __init__(self):
        self.obs = []
        self.acs = []
        self.obs_ = []
        self.obs_shape = None
        self.labels = []

def get_dataloader(opt, train=True, domain='source'):
    os.makedirs(opt.data_path, exist_ok=True)
    transform = [transforms.Resize(opt.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
    if domain == 'target':
        # inverting pixel values
        transform += [transforms.Lambda(lambda x: -1*x)]
    transform = transforms.Compose(transform)

    dataloader = torch.utils.data.DataLoader(
        datasets.MNIST(
            opt.data_path,
            train=train,
            download=True,
            transform=transform,
        ),
        batch_size=opt.batch_size,
        shuffle=True,
    )
    return dataloader
    
def dataloader_to_obs(dataloader):
    imgs = []
    labels = []
    for img, label in dataloader:
        imgs.append(img)
        labels.append(label)
    obs, labels = torch.cat(imgs), torch.cat(labels)
    obs = obs.type(torch.FloatTensor)
    return obs, labels

label_to_ac = {
    0: 1,
    1: 2,
    2: 0,
    3: 1,
    4: 2,
    5: 0,
    6: 1,
    7: 0,
    8: 2,
    9: 1
}

def gen_data(opt, train, domain):
    dataloader = get_dataloader(opt, train, domain)
    data = Data()
    data.obs, data.labels = dataloader_to_obs(dataloader)
    data.obs_, data.acs = data.obs.clone(), data.labels.clone()
    for i in range(data.obs_.shape[0]):
        data.acs[i] = label_to_ac[data.acs[i].item()]
        data.obs_[i, 0] = torch.tensor(np.rot90(data.obs_[i, 0].cpu().numpy(), data.acs[i].item()-1).copy())
    return data

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
    parser.add_argument("--channels", type=int, default=1, help="number of image channels")
    parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
    parser.add_argument("--download_path", default='./data/mnist_raw', help="data will be downloaded into this folder")
    parser.add_argument("--data_path", default='./data/mnist_prep', help="path to processed data")
    parser.add_argument('--seed', type=int, default=0, help='random seed (default: 0)')
    opt = parser.parse_args()
    
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    
    train_data_s = gen_data(opt, train=True, domain='source')
    train_data_t = gen_data(opt, train=True, domain='target')
    test_data_s = gen_data(opt, train=False, domain='source')
    test_data_t = gen_data(opt, train=False, domain='target')
    
    joblib.dump([train_data_s, train_data_t, test_data_s, test_data_t], os.path.join(opt.data_path, 'all_data.pkl'))