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
def get_the_date(Image_size,Image_path,batch_size):
    """ get the anime faces data """

    # apply the transformation
    TRANSFORM_IMG=transforms.Compose([transforms.ToTensor(),transforms.Resize(Image_size),transforms.RandomHorizontalFlip(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    # get the training data from the folder
    x_train = torchvision.datasets.ImageFolder(root=Image_path, transform=TRANSFORM_IMG)
    # initialize the loader
    loader = torch.utils.data.DataLoader(x_train, batch_size=batch_size, shuffle=True,  num_workers=1)
    return loader