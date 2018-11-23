from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import scipy.io as sio
import convLSTM_classifier

outfeat = sio.loadmat('output.mat')
outfeat = torch.tensor(outfeat['out'][:95])

device = torch.device("cpu")

model = convLSTM_classifier.convLSTMClassifier()

model = model.to(device)
model.eval()
outfeat = outfeat.float()
output = model.forward(outfeat)

print(output.size())
