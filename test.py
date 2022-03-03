import 
from turtle import forward


import torch
import torch.nn as nn


class K(nn.Module):
    def __init__(self) -> None:
        super(K, self).__init__()
        self.a = nn.parameter(2)

    def forward(self, x):
        return x * self.a

if __name__ == '__main__':

    model = K()
    optim = torch.optim.SGD(K.parameters)
    x = torch.ones(4,4)
    x = model(x)
    loss = x.sum()
    loss.backward()
    