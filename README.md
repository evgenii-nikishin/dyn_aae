# Domain Adaptation with Shared Latent Dynamics and Adversarial Matching
This repository contains PyTorch code for the paper

[Unsupervised Domain Adaptation with Shared Latent Dynamics for Reinforcement Learning](https://drive.google.com/open?id=1r7g4fbMDd55jMhEUVhDWYwXEjPyVLG2t) (accepted to [Bayesian Deep Learning workshop @ NeurIPS 2019](http://bayesiandeeplearning.org/))

by Evgenii Nikishin, Arsenii Ashukha and Dmitry Vetrov.


# Introduction

Reinforcement learning algorithms struggle to adapt quickly to new environments. We propose a model that learns similar latent representations for
similar pairs of observations from different domains without access to a one-to-one correspondence between the observations.
The model uses shared dynamics in a latent space
and adversarial matching of latent codes as a way to align latent representations.
Given the aligned latent space, the model aims to learn a policy upon the latent representations that is optimal for both of the environments. An illustration of effects of introducing shared dynamics and the adversarial loss can be found below:

<p align="center">
  <img src="https://user-images.githubusercontent.com/14283069/64994599-0960fd00-d8a7-11e9-9edb-e5839ce90623.png" width=500>
</p>

# Dependencies
* [PyTorch](http://pytorch.org/)
* [torchvision](https://github.com/pytorch/vision/)


# Usage example

To reproduce the results from the paper run 
```bash
python gen_data.py  
python train.py --domain=source  # Train encoding and a policy on source
python train.py --domain=target --seed=2  # Train encoding on target, we got the best results with seed 2
```


# Results

## MNIST with rotations

We design an artificial environment with MNIST digits as observations and rotations (-90, 0 and +90 degrees) as actions. After randomly assigning correct actions to all 10 digits, +1 reward is given for a correct action and 0 otherwise.
To obtain the target environment, we invert the pixel values of all images in the dataset.

To demonstrate that the representations learned for different domains are aligned, we use a decoder for one domain to reconstruct latent codes produced by an encoder for another domain:

<p align="center">
<img src="https://user-images.githubusercontent.com/14283069/64994601-0bc35700-d8a7-11e9-8832-1f59632f65cf.png" width=500>
</p>


We compare policies trained upon the learned latent representations using data from source and report reward on the target environment:

| Const prediction             |  VAE features     | Adversarial only | Dynamics only | Dynamics+Adversarial |
| ------------------------- |:------------:|:------------:|:----------------:|:---------------:|
| 0.40 ± 0 | 0.40 ± 0.03 | 0.45 ± 0.06 | 0.54 ± 0.07 | 0.81 ± 0.21 |


# References
 
 The code is adapted from
 * [github.com/eriklindernoren/PyTorch-GAN](https://github.com/eriklindernoren/PyTorch-GAN/tree/master/implementations/aae)
