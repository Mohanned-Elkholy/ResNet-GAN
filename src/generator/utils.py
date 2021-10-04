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



class ResBlockGenerator(nn.Module):
    """ This class make the standard resblock generator """

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlockGenerator, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1)
        nn.init.xavier_uniform(self.conv1.weight.data, 1.)
        nn.init.xavier_uniform(self.conv2.weight.data, 1.)

        self.model = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            self.conv1,
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            self.conv2
            )
        
        self.shortcut = nn.Sequential()

        if stride != 1:
            self.shortcut = nn.Upsample(scale_factor=2)

    def forward(self, x):
        return self.model(x) + self.shortcut(x)

class Dense_block(nn.Module):
    """ This is the initial dense block as in the paper """
    def __init__(self,in_channels,out_channels):
        super(Dense_block, self).__init__()

        self.Dense = torch.nn.Linear(in_channels,out_channels)
        nn.init.xavier_uniform(self.Dense.weight.data, 1.)
        self.activation = torch.nn.LeakyReLU(0.2)


    def forward(self,x):
        return self.activation(self.Dense(x))




class Generator(nn.Module):
    def __init__(self, z_dim,GEN_SIZE, channels,Image_size,initial_width=4,num_initial_dense=4):
        super(Generator, self).__init__()
        self.z_dim = z_dim
        self.GEN_SIZE=GEN_SIZE
        # initial width of the generator
        self.initial_width = self.initial_height = initial_width
        # the first dense layer
        self.dense = nn.Linear(self.z_dim, self.initial_width * self.initial_height * GEN_SIZE)
        # the first dense blocks
        self.initial_dense_blocks = [Dense_block(self.initial_width * self.initial_height * GEN_SIZE,self.initial_width * self.initial_height * GEN_SIZE) for i in range(num_initial_dense)]
        # mapping stack
        self.mapping_stack = nn.Sequential(*self.initial_dense_blocks)
        # the final conv layer
        self.final = nn.Conv2d(GEN_SIZE, channels, 3, stride=1, padding=1)
        nn.init.xavier_uniform(self.dense.weight.data, 1.)
        nn.init.xavier_uniform(self.final.weight.data, 1.)


        # number of Resblocks
        res_block_number = int(np.log(Image_size)/np.log(2))-int(np.log(self.initial_width)/np.log(2))
        # make the list of the initial resblocks
        self.generator_blocks = [ResBlockGenerator(GEN_SIZE, GEN_SIZE, stride=2) for i in range(res_block_number)]
        # the model
        self.model = nn.Sequential(
            # the initial dense and the reshaping layer are in the forward
            
            *self.generator_blocks,
            nn.BatchNorm2d(GEN_SIZE),
            nn.ReLU(),
            self.final,
            nn.Tanh()

            )

    def forward(self, z):
        # initial z
        z = self.dense(z)
        # mapping
        z = self.mapping_stack(z)
        # reshape
        z = z.view(-1, self.GEN_SIZE, self.initial_width, self.initial_height)
        # get the output
        output = self.model(z)

        return output
