import torch
from torch import nn
from torch.nn import functional as F


class VisionContextAdapter(nn.Module):
    def __init__(self, c_in, bottleneck=768):
        super(VisionContextAdapter, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(c_in, bottleneck, bias=False),
            nn.LeakyReLU(inplace=False)
        )

    def forward(self, x):
        x = self.fc1(x)
        return x



class TextAdapter(nn.Module):
    def __init__(self, c_in, bottleneck=768):
        super(TextAdapter, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(c_in, bottleneck, bias=False),
            nn.LeakyReLU(inplace=False)
        )

    def forward(self, x):
        x = self.fc1(x)
        return x
