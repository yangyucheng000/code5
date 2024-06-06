import mindspore as torch
import mindspore.nn as nn
from mindspore import ops as F
import math
from my_pooling import my_MaxPool2d, my_AvgPool2d


class Bottleneck(nn.Cell):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, pad_mode="pad",
                               padding=1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, pad_mode="pad")
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU()
        self.downsample = downsample
        self.stride = stride

    def construct(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNetBackbone(nn.Cell):

    def __init__(self, lda_out_channels, in_chn, block, layers, num_classes=1000):
        super(ResNetBackbone, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, pad_mode="pad", padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode="pad", padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.perceptual_conv1 = nn.SequentialCell(
            nn.Conv2d(256, 256, kernel_size=1, stride=1, pad_mode="pad", padding=0),
            nn.AvgPool2d(8, stride=8),
        )
        self.perceptual_conv2 = nn.SequentialCell(
            nn.Conv2d(512, 256, kernel_size=1, stride=1, pad_mode="pad", padding=0),
            nn.AvgPool2d(4, stride=4),
        )
        self.perceptual_conv3 = nn.SequentialCell(
            nn.Conv2d(1024, 256, kernel_size=1, stride=1, pad_mode="pad", padding=0),
            nn.AvgPool2d(2, stride=2),
        )
        self.perceptual_conv4 = nn.SequentialCell(
            nn.Conv2d(2048, 256, kernel_size=1, stride=1, pad_mode="pad", padding=0),
            nn.AvgPool2d(1, stride=1),
        )

        self.fc_conv = nn.SequentialCell(
            nn.Conv2d(1024, 256, kernel_size=1, stride=1, pad_mode="pad", padding=0),
            nn.AdaptiveAvgPool2d((1, 1))
        )

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.SequentialCell(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, pad_mode="pad"),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.SequentialCell(*layers)

    def construct(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x1 = x

        # the same effect as lda operation in the paper, but save much more memory
        # lda_1 = self.perceptual_conv1(x)
        x = self.layer2(x)
        x2 = x

        # lda_2 = self.perceptual_conv2(x)
        x = self.layer3(x)
        x3 = x

        # lda_3 = self.perceptual_conv3(x)
        x = self.layer4(x)
        x4 = x

        # lda_4 = self.perceptual_conv4(x)

        # vec = torch.cat((lda_1, lda_2, lda_3, lda_4), 1)
        #
        # vec = self.fc_conv(vec)
        # vec = vec.view(vec.size(0), -1)
        # out = self.fc_out(vec)

        return x1, x2, x3, x4


def resnet50_backbone(lda_out_channels, in_chn, pretrained=False, **kwargs):
    """Constructs a ResNet-50 model_hyper.
    Args:
        pretrained (bool): If True, returns a model_hyper pre-trained on ImageNet
    """
    model = ResNetBackbone(lda_out_channels, in_chn, Bottleneck, [3, 4, 6, 3], **kwargs)
    # if pretrained:
    #     save_model = model_zoo.load_url('https://download.pytorch.org/models/resnet50-19c8e357.pth')
    #     model_dict = model.state_dict()
    #     state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
    #     model_dict.update(state_dict)
    #     model.load_state_dict(model_dict)
    # else:
    #     model.apply(weights_init_xavier)
    return model


def supervisor(x, cnum):
    branch = x
    average = my_AvgPool2d(kernel_size=(1, branch.shape[1]), stride=(1, branch.shape[1]))(branch)

    branch = -my_MaxPool2d(kernel_size=(1, cnum), stride=(1, cnum))(-branch)
    shift = (branch - average)**2
    loss_2 = 1.0/(torch.exp(torch.mean(shift)))
    # branch = x
    # # branch = my_MaxPool2d(kernel_size=(1, 16), stride=(1, 16))(branch)
    #
    # branch = branch.reshape(branch.size(0), branch.size(1), branch.size(2) * branch.size(3))
    # branch = F.softmax(branch, 2)
    # branch = branch.reshape(branch.size(0), branch.size(1), x.size(2), x.size(2))
    # branch = my_MaxPool2d(kernel_size=(1, cnum), stride=(1, cnum))(branch)h.size(3))
    #     # branch = F.softmax(branch, 2)
    # branch = branch.reshape(branch.size(0), branch.size(1), branch.size(2) * branc
    # branch = branch.reshape(branch.size(0), branch.size(1), x.size(2), x.size(2))
    # branch = nn.MaxPool2d(7, stride=1)(branch)
    # # branch = my_MaxPool2d(kernel_size=(1, 64), stride=(1, 64))(branch)
    #
    # loss_2 = 1.0 - 1.0 * torch.mean(branch) / 49  # set margin = 3.0
    return loss_2


def CosSimilarity(lda_1, lda_2, lda_3, lda_4):
    loss = 0
    B, _, _, _ = lda_1.shape

    lda_1 = lda_1.reshape(lda_1.size(0), lda_1.size(1), lda_1.size(2) * lda_1.size(3))
    lda_2 = lda_2.reshape(lda_2.size(0), lda_2.size(1), lda_2.size(2) * lda_2.size(3))
    lda_3 = lda_3.reshape(lda_3.size(0), lda_3.size(1), lda_3.size(2) * lda_3.size(3))
    lda_4 = lda_4.reshape(lda_4.size(0), lda_4.size(1), lda_4.size(2) * lda_4.size(3))

    lda_1_sum = 1-torch.stack([F.normalize(lda_1[i]).mm(F.normalize(lda_1[i].t())) for i in range(B)]).float()
    lda_2_sum = 1-torch.stack([F.normalize(lda_2[i]).mm(F.normalize(lda_2[i].t())) for i in range(B)]).float()
    lda_3_sum = 1-torch.stack([F.normalize(lda_3[i]).mm(F.normalize(lda_3[i].t())) for i in range(B)]).float()
    lda_4_sum = 1-torch.stack([F.normalize(lda_4[i]).mm(F.normalize(lda_4[i].t())) for i in range(B)]).float()

    vec = torch.cat((lda_1_sum.unsqueeze(1), lda_2_sum.unsqueeze(1), lda_3_sum.unsqueeze(1), lda_4_sum.unsqueeze(1)), 1)
    loss = vec.mean(1).sum()/(B*256*256)

    return loss


class TargetNet(nn.Cell):
    """
    Target network for quality prediction.
    """
    def __init__(self, lda_out_channels, in_chn, pretrained=False):
        super(TargetNet, self).__init__()

        model = ResNetBackbone(lda_out_channels, in_chn, Bottleneck, [3, 4, 6, 3])
        self.res = model

        self.branch1 = Branch()
        self.branch2 = Branch()

    def construct(self, x):
        x1, x2, x3, x4 = self.res(x)
        out1 = self.branch1(x1, x2, x3, x4)
        out2 = self.branch2(x1, x2, x3, x4)

        return out1, out2


class Branch(nn.Cell):
    def __init__(self):
        super(Branch, self).__init__()
        self.perceptual_conv1 = nn.SequentialCell(
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0, has_bias=False, group=64),
            nn.AvgPool2d(8, stride=8),
        )
        self.perceptual_conv2 = nn.SequentialCell(
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0, has_bias=False, group=64),
            nn.AvgPool2d(4, stride=4),
        )
        self.perceptual_conv3 = nn.SequentialCell(
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0, has_bias=False, group=64),
            nn.AvgPool2d(2, stride=2),
        )
        self.perceptual_conv4 = nn.SequentialCell(
            nn.BatchNorm2d(2048),
            nn.ReLU(),
            nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0, has_bias=False, group=64),
            nn.AvgPool2d(1, stride=1),
        )

        self.fc_conv = nn.SequentialCell(
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=0, has_bias=False, group=256),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.fc_in = nn.Dense(256, 256)
        self.sigmoid = nn.Sigmoid()
        self.drop = nn.Dropout(0.5)
        self.gelu = nn.GELU()
        self.prelu = nn.PReLU()
        self.relu = nn.ReLU()
        self.fc_out = nn.Dense(256, 1)

    def construct(self, x1, x2, x3, x4):
        lda_1 = self.perceptual_conv1(x1)
        lda_2 = self.perceptual_conv2(x2)
        lda_3 = self.perceptual_conv3(x3)
        lda_4 = self.perceptual_conv4(x4)
        vec = F.cat((lda_1, lda_2, lda_3, lda_4), 1)

        # MC_loss = supervisor(vec, cnum=4)
        vec = my_MaxPool2d(kernel_size=(1, 4), stride=(1, 4))(vec)

        # MC_loss = 0
        # con_loss = CosSimilarity(lda_1, lda_2, lda_3, lda_4)

        vec = self.fc_conv(vec)
        vec = vec.view(vec.size(0), -1)

        vec = self.relu(vec)
        vec = self.fc_in(vec)
        vec = self.drop(vec)
        vec = self.relu(vec)
        out = self.fc_out(vec)

        return out