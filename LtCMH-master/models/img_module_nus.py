import torch
from torch import nn
from torch.nn import functional as F
from models.basic_module import BasicModule

LAYER1_NODE = 8192


def weights_init(m):
    if type(m) == nn.Conv2d:
        nn.init.normal_(m.weight.data, 0.0, 0.01)
        nn.init.normal_(m.bias.data, 0.0, 0.01)


class ImgModuleNus(BasicModule):
    def __init__(self, y_dim, bit):
        """
        :param y_dim: dimension of tags
        :param bit: bit number of the final binary code
        """
        super(ImgModuleNus, self).__init__()
        self.module_name = "img_module_nus"

        # full-conv layers
        self.conv1 = nn.Conv2d(1, LAYER1_NODE, kernel_size=(y_dim, 1), stride=(1, 1))
        self.conv2 = nn.Conv2d(LAYER1_NODE, bit, kernel_size=1, stride=(1, 1))
        self.apply(weights_init)
        self.e1 = nn.Sequential(
            nn.Linear(bit, bit),
            nn.Tanh()
        )
        self.e2 = nn.Sequential(
            nn.Linear(bit, bit),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = x.squeeze()
        return x

