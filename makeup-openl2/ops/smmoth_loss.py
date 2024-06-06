import numpy as np

import mindspore as ms
from mindspore import ops
from mindspore import nn

import os


def to_var(x, requires_grad=True):
    if requires_grad:
        #TODO Variable
        # return Variable(x).float()
        return ms.Tensor(x).float()
    else:
        # return Variable(x, requires_grad=requires_grad).float()
        return ms.Tensor(x).float()

class SmmothLoss(nn.Cell):
    def __init__(self):
        super(SmmothLoss, self).__init__()



    def construct(self, input_data, mask_tar):
        mask_tar=mask_tar.squeeze(0)
        mask_tar=mask_tar.broadcast_to(1, 3, mask_tar.size(2), mask_tar.size(2)).squeeze()
        #input_crop=input_data*mask_tar
        kernel = ms.Tensor(np.broadcast_to(np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]]), (input_data.size(0), input_data.size(1), 3, 3)),
                              dtype=ms.float32)
        laplace_input=ops.conv2d(input_data, kernel, bias=None, stride=1, pad_mode='pad', padding=1, dilation=1, groups=1)
        laplace_input=laplace_input*mask_tar
        input_zeros=ops.zeros_like(laplace_input)
        input_zeros=to_var(input_zeros,requires_grad=False)
        local_transfer_loss = ops.l1_loss(laplace_input,input_zeros) * 1
        return local_transfer_loss





# class SmmothLoss_skin(nn.Cell):
#     def __init__(self):
#         super(SmmothLoss_skin, self).__init__()


#     def forward(self, input_data, tar_data,mask_tar):
#         mask_tar=mask_tar.squeeze(0)
#         mask_tar=mask_tar.expand(1, 3, mask_tar.size(2), mask_tar.size(2)).squeeze()
#         kernel = torch.tensor(np.broadcast_to(np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]]), (input_data.size(0), input_data.size(1), 3, 3)),dtype=torch.float32).to("cuda")
#         laplace_input=F.conv2d(input_data, kernel, bias=None, stride=1, padding=1, dilation=1, groups=1)
#         laplace_tar = F.conv2d(tar_data, kernel, bias=None, stride=1, padding=1, dilation=1, groups=1)
#         input_data=input_data.squeeze()
#         tar_data = tar_data.squeeze()
#         laplace_input = input_data * mask_tar
#         laplace_tar=tar_data*mask_tar
#         #img_train_list = torch.cat([laplace_input,laplace_tar],dim=2)
#         #save_path = os.path.join('/opt/tiger/sjn_makeup','1.jpg')
#         #save_image(img_train_list.data,save_path, normalize=True)
#         #laplace_input=torch.zeros_like(input_data,device='cpu')
#         #laplace_input=input_data[:,:,int(rect[1]):int(rect[3]),int(rect[0]):int(rect[2])]
#         #laplace_input=input_data[:,:,int(rect[5]):int(rect[7]),int(rect[4]):int(rect[6])]
#         #lapalce_tar = torch.zeros_like(input_data,device='cpu')
#         #laplace_tar = tar_data[:,:,int(rect[1]):int(rect[3]),int(rect[0]):int(rect[2])]
#         #laplace_tar = tar_data[:,:,int(rect[5]):int(rect[7]),int(rect[4]):int(rect[6])]
#         local_transfer_loss = F.l1_loss(laplace_input,laplace_tar) 
#         return local_transfer_loss

