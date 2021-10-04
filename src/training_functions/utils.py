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

def train_one_epoch(loader,batch_size,disc_iters,Z_dim,optim_disc,optim_gen,discriminator,generator,epoch=0):
    """ This function applies one forward propagation """
    for (data,_) in tqdm(loader):
        if data.size()[0] != batch_size:
            continue

        # convert the data to a torch variable 
        data = Variable(data.cuda())
        data = data.type(torch.FloatTensor)
        data = data.cuda()


        #############################################################################################################
        #### Discriminator
        #############################################################################################################

        # iterate over the number of discriminator updates
        for _ in range(disc_iters):
            z = Variable(torch.randn(batch_size, Z_dim).cuda())
            # zero initialize the gradients
            optim_disc.zero_grad()
            optim_gen.zero_grad()

            # get the loss of the discriminator
            disc_loss = nn.ReLU()(1.0 - discriminator(data)).mean() + nn.ReLU()(1.0 + discriminator(generator(z))).mean()

            # optimize
            disc_loss.backward()
            optim_disc.step()


        #############################################################################################################
        #### Generator
        #############################################################################################################

        # get a random variable
        z = Variable(torch.randn(batch_size, Z_dim).cuda())

        # zero initialize the optimizers
        optim_disc.zero_grad()
        optim_gen.zero_grad()

        # get the generator loss
        gen_loss = -1*discriminator(generator(z)).mean()

        # optimize
        gen_loss.backward()
        optim_gen.step()

        # print losses
    print(f"Epoch: {epoch}" , 'disc loss', disc_loss.item(), 'gen loss', gen_loss.item())
