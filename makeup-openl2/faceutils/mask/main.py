#!/usr/bin/python
# -*- encoding: utf-8 -*-
import os.path as osp

import numpy as np
import cv2
from PIL import Image

import mindspore as ms
import mindspore.dataset.vision as vision
import mindspore.dataset.transforms as transforms
from mindspore import ops
import mindspore.nn as nn

from .model import BiSeNet


class FaceParser:
    def __init__(self, device="cpu"):
        mapper = [0, 1, 2, 3, 4, 5, 0, 11, 12, 0, 6, 8, 7, 9, 13, 0, 0, 10, 0]
        self.device = device
        self.dic = ms.Tensor(mapper).unsqueeze(1)
        save_pth = osp.split(osp.realpath(__file__))[0] + '/resnet.ckpt'

        net = BiSeNet(n_classes=19)
        para_dict = ms.load_checkpoint(save_pth)
        ms.load_param_into_net(net,para_dict)
        # net.load_state_dict(torch.load(save_pth, map_location=device))
        self.net = net.set_train(False)
        self.to_tensor = transforms.Compose([
            vision.ToTensor(),
            vision.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225), is_hwc=False),
        ])


    def parse(self, image: Image):
        assert image.shape[:2] == (512, 512)
        # with torch.no_grad():
        image = self.to_tensor(image)
        image = ms.Tensor(image)
        image = ops.unsqueeze(image, 0)
        out = self.net(image)[0]
        parsing = out.squeeze(0).argmax(0)
        embedding = nn.Embedding(self.dic.shape[0],self.dic.shape[1],embedding_table=self.dic)
        parsing = embedding(parsing)
        # parsing = torch.nn.functional.embedding(parsing, self.dic)
        return parsing.float()

