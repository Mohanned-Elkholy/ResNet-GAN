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
import argparse

from keras.applications.inception_v3 import InceptionV3
from skimage.transform import resize
from keras.applications.inception_v3 import preprocess_input
from numpy import cov
from numpy import trace
from numpy import iscomplexobj
import numpy
from scipy.linalg import sqrtm

from src.data_manipulation_functions.utils import *
from src.discriminator.utils import *
from src.generator.utils import *
from src.helper_functions.utils import *
from src.training_functions.utils import *

# !pip install argparse

#number of updates to discriminator for every update to generator 

parser = argparse.ArgumentParser()

parser.add_argument('--dataset_path',type=str, required = True)
parser.add_argument('--GAN_check_point_path',type=str,required = True)
parser.add_argument('--outputs',type=str,required = True)
parser.add_argument('--continue_training_from_checkpoint',type=bool,required = True)

opt = parser.parse_args()


batch_size=10

lr=2e-4

loss='hinge'

Z_dim = 128

disc_iters = 2

channels = 3 # number of final channels

GEN_SIZE = 128 # the size of channels along the generator

DISC_SIZE = 128 # the size of channels along the discriminator

Image_size = 256 # the size of images

continue_training_from_checkpoint = False

epochMaxNumber = 3000

GAN_check_point = opt.GAN_check_point_path


Image_path = opt.dataset_path

loader = get_the_date(Image_size,Image_path,batch_size)
Store_progress_plot_images_path = opt.outputs

# get the generator and the discriminator
discriminator = Discriminator(channels,DISC_SIZE).cuda()
generator = Generator(Z_dim,GEN_SIZE, channels,Image_size).cuda()

# set up the optimizers
optim_disc = optim.Adam(filter(lambda p: p.requires_grad, discriminator.parameters()), lr=lr, betas=(0.0,0.9))
optim_gen  = optim.Adam(generator.parameters(), lr=lr, betas=(0.0,0.9))

epoch_number = 0
FID_list=[]
fixed_z = Variable(torch.randn(batch_size, Z_dim).cuda())
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
    epoch_number = GAN_checkpoint_dictionary['epoch_number']
    fixed_z = GAN_checkpoint_dictionary['fixed_Z_vector']

get_FID_every = 10

for epoch in range(epoch_number,epochMaxNumber):
    plot_outputs(generator,fixed_z,Store_progress_plot_images_path,epoch)
    train_one_epoch(loader,batch_size,disc_iters,Z_dim,optim_disc,optim_gen,discriminator,generator,epoch)

