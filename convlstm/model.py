import torch.nn as nn
from torchvision import models
from convlstm_cell import *


class ConvLSTMModel(nn.Module):
    def __init__(self, mem_size, img_split_type):
        super(ConvLSTMModel, self).__init__()
        self.resnet = models.resnet18(pretrained=False)
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.convnet = nn.Sequential(*list(self.resnet.children())[:-2])
        self.resnet = None
        self.convlstm = ConvLSTMCell(512, mem_size)

        self.maxpool = nn.MaxPool2d(2)
        self.relu = nn.ReLU()

        if img_split_type == 2:
            self.lin1 = nn.Linear(2 * 21 * 2 * mem_size, 1000)
        else:
            self.lin1 = nn.Linear(7 * 7 * 2 * mem_size, 1000)
        self.lin2 = nn.Linear(1000, 256)
        self.lin3 = nn.Linear(256, 1)

        self.classifier = nn.Sequential(self.lin1, self.relu, self.lin2, self.relu, self.lin3)

    def forward(self, x):
        x = x.permute(1, 0, 2, 3, 4)
        x_state1 = None
        x_state2 = None
        seq_len = x.size(0)

        for t in range(0, seq_len):
            # forward
            x1 = self.convnet(x[t])
            x_state1 = self.convlstm(x1, x_state1)
            # backward
            x2 = self.convnet(x[seq_len - (t + 1)])
            x_state2 = self.convlstm(x2, x_state2)

        state = torch.cat((x_state1[0], x_state2[0]), axis=1)
        state = self.maxpool(state)
        state = state.view(state.size(0), -1)

        state = self.classifier(state)
        return state