import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
from torchvision import datasets, transforms
from torch.autograd import Variable
from torch import nn
import torch.nn.functional as F
from tqdm.notebook import tqdm

import torch.nn.utils.spectral_norm as SpectralNorm
import numpy as np

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
from torchvision import datasets, transforms
from torch.autograd import Variable
from torch import nn
import torch.nn.functional as F

import torch.nn.utils.spectral_norm as SpectralNorm
import numpy as np

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import torchvision
from IPython.display import clear_output 

from keras.applications.inception_v3 import InceptionV3
from skimage.transform import resize
from keras.applications.inception_v3 import preprocess_input
from numpy import cov
from numpy import trace
from numpy import iscomplexobj
import numpy
from scipy.linalg import sqrtm

def save_gan_check_point(saving_path):
    """ this function saves the checkpoint in the given paths """
    torch.save({
            'epoch_number': epoch_number,
            'generator_state_dict': generator.state_dict(),
            'discriminator_state_dict': discriminator.state_dict(),

            'optimizer_G_state_dict': optim_gen.state_dict(),
            'optimizer_D_state_dict': optim_disc.state_dict(),

            'FID_list': FID_list,
            'fixed_Z_vector': fixed_z,

            }, saving_path)


def generate_models(continue_training_from_checkpoint=False):
    """ 
    this function produces the generator and the discriminator, it also uploads the checkpoints
    
    outputs
    generator: generator model
    optim_gen: generator's optimizaer
    discriminator: discriminator model
    optim_disc: discriminator's optimizer

    """
    # models
    discriminator = Discriminator().cuda()
    generator = Generator(Z_dim).cuda()

    # their corrosponding optimizers
    optim_disc = optim.Adam(filter(lambda p: p.requires_grad, discriminator.parameters()), lr=lr, betas=(0.0,0.9))
    optim_gen  = optim.Adam(generator.parameters(), lr=lr, betas=(0.0,0.9))

    # initialize variables
    epoch_number = 0
    FID_list=[]
    fixed_z = Variable(torch.randn(batch_size, Z_dim).cuda())

    # check
    if continue_training_from_checkpoint:
        #load dictionary
        GAN_checkpoint_dictionary = torch.load(GAN_check_point)
        # set up the epoch number
        generator.load_state_dict(GAN_checkpoint_dictionary['generator_state_dict'])
        discriminator.load_state_dict(GAN_checkpoint_dictionary['discriminator_state_dict'])
        # set up the optimizer states
        optim_disc.load_state_dict(GAN_checkpoint_dictionary['optimizer_D_state_dict'])
        optim_gen.load_state_dict(GAN_checkpoint_dictionary['optimizer_G_state_dict'])
        # set up training parameters
        FID_list = GAN_checkpoint_dictionary['FID_list']
        epoch_number = GAN_checkpoint_dictionary['epoch_number']
        fixed_z = GAN_checkpoint_dictionary['fixed_Z_vector']

    return generator,optim_gen,discriminator,optim_disc


def plot_outputs(generator,fixed_z,save_path,epoch):
    samples = generator(fixed_z).cpu().data.numpy()[:4]
    fig = plt.figure(figsize=(64, 64))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples[:4]):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.transpose((1,2,0)) * 0.5 + 0.5)

    plt.savefig(save_path+'/{}.png'.format(str(epoch).zfill(3)), bbox_inches='tight')
    plt.close(fig)
