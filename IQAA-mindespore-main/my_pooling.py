import mindspore
import numpy as np
import random
from mindspore.nn.grad import Vjp
import mindspore.nn as nn
# from torch.nn.modules.utils import _single, _pair, _triple
import mindspore.ops as F
# from torch.nn.parameter import Parameter



class my_MaxPool2d(nn.Cell):


    def __init__(self, kernel_size, stride=None, padding=0, dilation=1,
                 return_indices=False, ceil_mode=False):
        super(my_MaxPool2d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride or kernel_size
        self.padding = padding
        self.dilation = dilation
        self.return_indices = return_indices
        self.ceil_mode = ceil_mode

    def construct(self, input):
        # input = input.transpose
        input = F.transpose(input, (0, 3, 2, 1))


        input = F.max_pool2d(input, self.kernel_size, self.stride,
                            self.padding, self.dilation, self.ceil_mode,
                            self.return_indices)
        # input = input.transpose(3,1).contiguous()
        input = F.transpose(input, (0, 3, 2, 1)).contiguous()

        return input

    # def __repr__(self):
    #     kh, kw = _pair(self.kernel_size)
    #     dh, dw = _pair(self.stride)
    #     padh, padw = _pair(self.padding)
    #     dilh, dilw = _pair(self.dilation)
    #     padding_str = ', padding=(' + str(padh) + ', ' + str(padw) + ')' \
    #         if padh != 0 or padw != 0 else ''
    #     dilation_str = (', dilation=(' + str(dilh) + ', ' + str(dilw) + ')'
    #                     if dilh != 0 and dilw != 0 else '')
    #     ceil_str = ', ceil_mode=' + str(self.ceil_mode)
    #     return self.__class__.__name__ + '(' \
    #         + 'kernel_size=(' + str(kh) + ', ' + str(kw) + ')' \
    #         + ', stride=(' + str(dh) + ', ' + str(dw) + ')' \
    #         + padding_str + dilation_str + ceil_str + ')'


class my_AvgPool2d(nn.Cell):
    def __init__(self, kernel_size, stride=None, padding=0, ceil_mode=False,
                 count_include_pad=True):
        super(my_AvgPool2d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride or kernel_size
        self.padding = padding
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad

    def construct(self, input):
        input = input.transpose(3,1)
        input = F.avg_pool2d(input, self.kernel_size, self.stride,
                            self.padding, self.ceil_mode, self.count_include_pad)
        input = input.transpose(3,1).contiguous()

        return input


    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'kernel_size=' + str(self.kernel_size) \
            + ', stride=' + str(self.stride) \
            + ', padding=' + str(self.padding) \
            + ', ceil_mode=' + str(self.ceil_mode) \
            + ', count_include_pad=' + str(self.count_include_pad) + ')'


# m = my_MaxPool2d((1, 32), stride=(1, 32))
# input = Variable(torch.randn(3, 2208, 7, 7))
# output = m(input)
# print(output.size())
