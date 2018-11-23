import torch
import resnetMod
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
from MyConvLSTMCell import *


class convLSTMClassifier(nn.Module):
    def __init__(self, num_classes=2, mem_size=512):
        super(convLSTMClassifier, self).__init__()
        self.num_classes = num_classes
        self.mem_size = mem_size
        self.lstm_cell = MyConvLSTMCell(512, mem_size)
        self.avgpool = nn.AvgPool2d(7)
        self.dropout = nn.Dropout(0.7)
        self.fc = nn.Linear(mem_size, self.num_classes)
        self.classifier = nn.Sequential(self.dropout, self.fc)

    def forward(self, inputVariable):
        state = (Variable(torch.zeros((inputVariable.size(1), self.mem_size, 7, 7))),
                 Variable(torch.zeros((inputVariable.size(1), self.mem_size, 7, 7))))
        for t in range(inputVariable.size(0)):
            m,n,o = inputVariable[t].size()
            inputs = inputVariable[t].reshape((1,m,n,o))
            state = self.lstm_cell(inputs, state)

        feats1 = self.avgpool(state[1]).view(state[1].size(0), -1)
        feats = self.classifier(feats1)
        return feats, feats1
