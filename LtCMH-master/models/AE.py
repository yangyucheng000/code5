import torch;
from torch import nn;

from config import opt
import numpy as np
import operator



class AutoEncoder(nn.Module):
    def __init__(self,**kwargs):
        super(AutoEncoder, self).__init__()
        opt.parse(kwargs)
        self.imgencoder=nn.Sequential(
            nn.Linear(opt.bit,opt.bit),
            nn.Tanh(),
            nn.Linear(opt.bit,opt.bit),
            nn.Tanh(),
            nn.Linear(opt.bit,opt.bit),
        )
        self.imgdecoder=nn.Sequential(
            nn.Linear(2*opt.bit,opt.bit),
            nn.Tanh(),
            nn.Linear(opt.bit,opt.bit),
            nn.Tanh(),
            nn.Linear(opt.bit,opt.bit),
            nn.Tanh(),
            nn.Linear(opt.bit,opt.bit),
            nn.Tanh()
        )
        self.txtencoder=nn.Sequential(
            nn.Linear(opt.bit,opt.bit),
            nn.Tanh(),
            nn.Linear(opt.bit ,opt.bit),
            nn.Tanh(),
            nn.Linear(opt.bit, opt.bit),

        )
        self.txtdecoder=nn.Sequential(
            nn.Linear(2 * opt.bit, opt.bit),
            nn.Tanh(),
            nn.Linear(opt.bit, opt.bit),
            nn.Tanh(),
            nn.Linear(opt.bit, opt.bit),
            nn.Tanh(),
            nn.Linear(opt.bit, opt.bit),
            nn.Tanh()
        )
        self.commonencoder = nn.Sequential(
            nn.Linear(2 * opt.bit, opt.bit),
            nn.Tanh(),
            nn.Linear(opt.bit, opt.bit),
            nn.Tanh(),
            nn.Linear(opt.bit, opt.bit),
            nn.Tanh(),
            nn.Linear(opt.bit, opt.bit),

        )

#还得结合共性做decoder
    def forward(self,inputsx,inputsy):

        imgindi= self.imgencoder(inputsx.clone())

        txtindi = self.txtencoder(inputsy.clone())

        common = self.commonencoder(torch.cat((imgindi,txtindi),dim=1))


        imgcon = torch.cat((imgindi, common), dim=1)
        txtcon = torch.cat((txtindi, common), dim=1)

        imgout = self.imgdecoder(imgcon.clone())
        txtout = self.txtdecoder(txtcon.clone())



        return imgindi,txtindi,common,imgout,txtout





