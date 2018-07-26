import torch


class ResNet(torch.nn.Module):
    def __init__(self, n=18):
        super(ResNet, self).__init__()
