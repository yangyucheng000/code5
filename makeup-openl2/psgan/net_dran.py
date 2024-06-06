#!/usr/bin/python
# -*- encoding: utf-8 -*-
import math

import mindspore as ms
import mindspore.nn as nn
from mindspore import ops
from mindcv.models import VGG as TVGG 

from ops.spectral_norm import spectral_norm as SpectralNorm
from concern.track import Track

import numpy as np
import os.path as osp
import os
# from torchvision.utils import save_image

# from torch.autograd import Variable
import cv2

class ResidualBlock_NoIn(nn.Cell):
    """Residual Block."""
    def __init__(self, dim_in, dim_out, net_mode=None):
        
        super(ResidualBlock_NoIn, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
        )
    def construct(self, x):
        return x + self.main(x)

class ResidualBlock(nn.Cell):
    """Residual Block."""
    def __init__(self, dim_in, dim_out, net_mode=None):
        if net_mode == 'p' or (net_mode is None):
            use_affine = True
        elif net_mode == 't':
            use_affine = False
        super(ResidualBlock, self).__init__()
        self.main = nn.SequentialCell(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, pad_mode='pad', padding=1, has_bias=False),
            nn.InstanceNorm2d(dim_out, affine=use_affine),
            nn.ReLU(),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, pad_mode='pad', padding=1, has_bias=False),
            nn.InstanceNorm2d(dim_out, affine=use_affine)
        )
    def construct(self, x):
        return x + self.main(x)

# def to_var(x, requires_grad=True):
#     if torch.cuda.is_available():
#         x = x.cuda()
#     if not requires_grad:
#         return Variable(x, requires_grad=requires_grad)
#     else:
#         return Variable(x)


class skin_layers(nn.Cell):
    def __init__(self,dim_in,dim_out):
        super(skin_layers,self).__init__()
        self.skin_gamma=nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=1, stride=1, padding=0, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True),
            nn.ReLU(inplace=True)
                )
        self.skin_beta=nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=1, stride=1, padding=0, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True),
            nn.ReLU(inplace=True)
               )
    def construct(self,x):
        return self.skin_gamma(x),self.skin_beta(x)

class GetMatrix(nn.Cell):
    def __init__(self, dim_in, dim_out):
        super(GetMatrix, self).__init__()
        self.get_gamma = nn.Conv2d(dim_in, dim_out, kernel_size=1, stride=1, padding=0, bias=False)
        self.get_beta = nn.Conv2d(dim_in, dim_out, kernel_size=1, stride=1, padding=0, bias=False)

    def construct(self, x):
        gamma = self.get_gamma(x)
        beta = self.get_beta(x)
        return x, gamma, beta


class Conv3_layer(nn.Cell):
    def __init__(self, dim_in, dim_out):
        super(Conv3_layer, self).__init__()

        self.conv1 = nn.Conv2d(dim_in, dim_in, kernel_size=3, stride=1, padding=0, bias=False)
        self.conv2 = nn.Conv2d(dim_in, dim_in*2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.InstanceNorm2d(2, affine=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(dim_in*2, dim_out, kernel_size=3, stride=1, padding=0, bias=False)

    def construct(self, x):
        out = self.conv1(x)
        return out




class NONLocalBlock2D(nn.Cell):
    def __init__(self):
        super(NONLocalBlock2D, self).__init__()
        self.g = nn.Conv2d(in_channels=1, out_channels=1,
                           kernel_size=1, stride=1, padding=0)

    # def construct(self, source, weight):
    #     """(b, c, h, w)
    #     src_diff: (3, 136, 32, 32)
    #     """
    #     batch_size = source.size(0)

    #     g_source = source.view(batch_size, 1, -1)  # (N, C, H*W)
    #     g_source = g_source.permute(0, 2, 1)  # (N, H*W, C)

    #     y = torch.bmm(weight.to_dense(), g_source)
    #     y = y.permute(0, 2, 1).contiguous()  # (N, C, H*W)
    #     y = y.view(batch_size, 1, *source.size()[2:])
    #     return y


def de_norm( x):
    out = (x + 1) / 2
    return out.clamp(0, 1)


# def vis_train(img_train_list, num):
#     # saving training results
#     # mode = "train_vis"
#     img_train_list = torch.cat(img_train_list, dim=3)
#     result_path_train = osp.join("/opt/tiger/sjn_makeup/", "align")
#     if not osp.exists(result_path_train):
#         os.makedirs(result_path_train)
#     save_path = os.path.join(result_path_train, str(num)+'.jpg')
#     save_image(de_norm(img_train_list.data), save_path, normalize=True)




def crop_face_best_fit(img_s, landmarks_s1,img_r,landmarks_r1, laplace_r):
    landmarks_s = np.ravel(landmarks_s1)
    landmarks_r =np.ravel(landmarks_r1)
    if len(landmarks_s)<1:
        return None,None
    affine_matrix = np.zeros((2, 3))
    affine_matrix_inverse = np.zeros((2, 3))
    src_vec, dst_vec = [], []
    mean_sx, mean_sy, mean_dx, mean_dy = 0.0, 0.0, 0.0, 0.0
    for i in range(int(len(landmarks_r) / 2)):
        mean_sx += landmarks_r[2 * i+1]
        mean_sy += landmarks_r[2 * i ]
        mean_dx += landmarks_s[2 * i+1]
        mean_dy += landmarks_s[2 * i]
    mean_sx /= (len(landmarks_r) / 2)
    mean_sy /= (len(landmarks_r) / 2)
    mean_dx /= (len(landmarks_s) / 2)
    mean_dy /= (len(landmarks_s) / 2)

    for i in range(int(len(landmarks_s) / 2)):
        src_vec.append(landmarks_r[2 * i+1] - mean_sx)
        src_vec.append(landmarks_r[2 * i] - mean_sy)
        dst_vec.append(landmarks_s[2 * i+1] - mean_dx)
        dst_vec.append(landmarks_s[2 * i] - mean_dy)
    a, b, norm = 0.0, 0.0, 0.0
    for i in range(int(len(landmarks_r) / 2)):
        a += src_vec[2 * i+1] * dst_vec[2 * i+1] + src_vec[2 * i ] * dst_vec[2 * i ]
        b += src_vec[2 * i+1] * dst_vec[2 * i ] - src_vec[2 * i ] * dst_vec[2 * i+1]
        norm += src_vec[2 * i+1] * src_vec[2 * i+1] + src_vec[2 * i] * src_vec[2 * i]

    a /= norm
    b /= norm
    mean_sx_trans = a * mean_sx - b * mean_sy
    mean_sy_trans = b * mean_sx + a * mean_sy
    affine_matrix[0, 0] =1.0/a
    affine_matrix[0, 1] = -b
    affine_matrix[1, 0] = b
    affine_matrix[1, 1] = 1.0/a
    affine_matrix[0, 2] = (mean_dx - mean_sx_trans)/img_r.shape[2]
    affine_matrix[1, 2] = (mean_dy - mean_sy_trans)/img_r.shape[3]

    #print("...............",img_s.shape)
    affine_matrix=ms.Tensor(affine_matrix,dtype=ms.float32)
    grid = ops.affine_grid(affine_matrix.unsqueeze(0), img_s.shape)
    grid=grid.float()
    output = ops.grid_sample(img_r, grid)
    output_laplace = ops.grid_sample(laplace_r, grid)
    
    return output, output_laplace
    



def judge_local(left_eye, right_eye, lms_s):
    rect_eye = [int(min(lms_s[0][:, 1])),
            int(min(lms_s[0][:, 0])),
            int(max(lms_s[0][:, 1])),
            int(max(lms_s[0][:, 0])),
            int(min(lms_s[1][:, 1])),
            int(min(lms_s[1][:, 0])),
            int(max(lms_s[1][:, 1])),
            int(max(lms_s[1][:, 0]))]
    rect_eye = np.array(rect_eye)
    w_left=int(rect_eye[3])-int(rect_eye[1])
    h_left=int(rect_eye[2]) - int(rect_eye[0])
    w_right=int(rect_eye[7])-int(rect_eye[5])
    h_right=int(rect_eye[6]) - int(rect_eye[4])
    flag = None
    area_left = ops.nonzero(left_eye).shape[0]
    area_right = ops.nonzero(right_eye).shape[0]
    if area_left == 0:
        return False
    if area_right == 0:
        return True

    if h_left*1 > h_right:
        return True
    if h_right*1 > h_left:
        return False
    if w_left*1 > w_right:
        return True
    if w_right*1 > w_left:
        return False
    
    if area_left  > area_right:
        return True
    if area_right  > area_left:
        return False
    return flag


def judge_local_brow(left_eye, right_eye, lms_s):
    flag=None
    
    area_left = ops.nonzero(left_eye).shape[0]
    area_right = ops.nonzero(right_eye).shape[0]
    if area_left  > area_right:
        return True
    if area_right  > area_left:
        return False
    return flag

def laplace_target_wrap(data_c, data_s, mask_c, mask_s, rect_c=None, rect_s=None, lms_c=None, lms_s=None, flag=True):
    if flag:
        erode_kernel = np.ones((2,2),np.uint8)
        mask_s_ballonly = mask_s[1].unsqueeze(0)
        mask_s_eyebrow = mask_s[2].unsqueeze(0)
        mask_s_skin = mask_s[3].unsqueeze(0)
        mask_s_ball = (mask_s[0]+mask_s[1])
        #TODO ms不支持 .to(device)
        mask_s_ball = ms.Tensor(cv2.erode(mask_s_ball.asnumpy().squeeze(),erode_kernel, iterations=1), dtype=ms.float32).unsqueeze(0).unsqueeze(0)  # .to("cuda")
        
        mask_s_np = mask_s_ball.squeeze().asnumpy()
        mask_s_np =  cv2.blur(mask_s_np,(21,21))
        mask_s_ball = ms.Tensor(mask_s_np).unsqueeze(0).unsqueeze(0)  # .to('cuda')
        mask_s = mask_s_ball * (1-(mask_s_ballonly+mask_s_eyebrow))*mask_s_skin

    if rect_c == None and rect_s == None:
        rect_c = [int(min(lms_c[0][:, 1])-8),
            int(min(lms_c[0][:, 0])-8),
            int(max(lms_c[0][:, 1])+9),
            int(max(lms_c[0][:, 0])+9),
            int(min(lms_c[1][:, 1])-8),
            int(min(lms_c[1][:, 0])-8),
            int(max(lms_c[1][:, 1])+9),
            int(max(lms_c[1][:, 0])+9)]
        rect_c = np.array(rect_c)
        rect_s = [int(min(lms_s[0][:, 1])-8),
            int(min(lms_s[0][:, 0])-8),
            int(max(lms_s[0][:, 1])+9),
            int(max(lms_s[0][:, 0])+9),
            int(min(lms_s[1][:, 1])-8),
            int(min(lms_s[1][:, 0])-8),
            int(max(lms_s[1][:, 1])+9),
            int(max(lms_s[1][:, 0])+9)]
        rect_s = np.array(rect_s)
    else:
        if rect_c.shape[0] != 1:
            rect_c = rect_c.asnumpy()
        else:
            rect_c = rect_c.squeeze(0).asnumpy()
        if rect_s.shape[0] != 1:
            rect_s = rect_s.asnumpy()
        else:
            rect_s = rect_s.squeeze(0).asnumpy()
    rect_c = rect_c/mask_c.shape[2] * data_s.shape[2]
    rect_s = rect_s/mask_s.shape[2] * data_s.shape[2]

    kernel = ms.Tensor(np.broadcast_to(np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]]), (data_s.shape[1], data_s.shape[1], 3, 3)),dtype=ms.float32)  # .to("cuda")
    laplace_c=ops.conv2d(data_c, kernel, bias=None, stride=1, pad_mode='pad', padding=1, dilation=1, groups=1)
    laplace_s = ops.conv2d(data_s, kernel, bias=None, stride=1, pad_mode='pad', padding=1, dilation=1, groups=1)
    laplace_c_ori= data_c * mask_c
    laplace_s_ori = data_s * mask_s
    laplace_s = laplace_s * mask_s
    laplace_c = laplace_c * mask_c

    s_w_left=int(rect_s[3])-int(rect_s[1])
    s_h_left=int(rect_s[2]) - int(rect_s[0])
    s_w_right=int(rect_s[7])-int(rect_s[5])
    s_h_right=int(rect_s[6]) - int(rect_s[4])

    c_w_left=int(rect_c[3])-int(rect_c[1])
    c_h_left=int(rect_c[2]) - int(rect_c[0])
    c_w_right=int(rect_c[7])-int(rect_c[5])
    c_h_right=int(rect_c[6]) - int(rect_c[4])

    #if the eyebrow didn't exist
    if not flag:
        index_brow = ops.nonzero(mask_s).asnumpy()
        index_brow_c = ops.nonzero(mask_c).asnumpy()
        brow_num = len(index_brow)
        brow_num_c = len(index_brow_c)
        lms_area = (s_w_left-17)*(s_h_left-17)+(s_w_right-17)*(s_h_right-17)
        lms_area_c = (c_w_left-17)*(c_h_left-17)+(c_w_right-17)*(c_h_right-17)
        if len(index_brow)<=0:
            return laplace_c, laplace_c_ori
        index_x_min, index_x_max = min(index_brow[:,2]), max(index_brow[:,2])
        index_y_min, index_y_max = min(index_brow[:,3]), max(index_brow[:,3])
        parsing_x = index_x_max-index_x_min
        parsing_y = index_y_max-index_y_min
        if (index_y_min * 0.8> rect_s[4] and index_y_max*1.2< rect_s[3]) or brow_num< lms_area*0.5 or brow_num_c < lms_area_c *0.5:
            return laplace_c, laplace_c_ori
            
    c_w_left=int(rect_c[3])-int(rect_c[1])
    c_h_left=int(rect_c[2]) - int(rect_c[0])
    c_w_right=int(rect_c[7])-int(rect_c[5])
    c_h_right=int(rect_c[6]) - int(rect_c[4])

    tar_w, tar_h = np.max(np.array([s_w_left,c_w_left, s_w_right, c_w_right]))+2, np.max(np.array([s_h_left, c_h_left, s_h_right, c_h_right]))+2
    tar_w=tar_w+tar_w%2
    tar_h=tar_h+tar_h%2

    center_c_left_x, center_c_left_y =int((rect_c[1]+rect_c[3])//2), int((rect_c[0]+rect_c[2])//2)
    center_c_right_x, center_c_right_y =int( (rect_c[5]+rect_c[7])//2), int((rect_c[4]+rect_c[6])//2)
    center_s_left_x, center_s_left_y =int((rect_s[1]+rect_s[3])//2), int((rect_s[0]+rect_s[2])//2)
    center_s_right_x, center_s_right_y =int( (rect_s[5]+rect_s[7])//2), int((rect_s[4]+rect_s[6])//2)

    rect_ori_c_left=laplace_c_ori[:,:,int(rect_c[1]):int(rect_c[3]), int(rect_c[0]):int(rect_c[2])]
    rect_ori_c_right=laplace_c_ori[:,:,int(rect_c[5]):int(rect_c[7]), int(rect_c[4]):int(rect_c[6])]

    rect_mask_c_left=mask_c[:,:,int(rect_c[1]):int(rect_c[3]), int(rect_c[0]):int(rect_c[2])]
    rect_mask_c_right=mask_c[:,:,int(rect_c[5]):int(rect_c[7]), int(rect_c[4]):int(rect_c[6])]
    if flag:
        rect_maskball_s_left=mask_s_ballonly[:,:,int(rect_c[1]):int(rect_c[3]), int(rect_c[0]):int(rect_c[2])]
        rect_maskball_s_right=mask_s_ballonly[:,:,int(rect_c[5]):int(rect_c[7]), int(rect_c[4]):int(rect_c[6])]

    rect_ori_s_left=laplace_s_ori[:,:,int(rect_s[1]):int(rect_s[3]), int(rect_s[0]):int(rect_s[2])]
    rect_laplace_s_left=laplace_s[:,:,int(rect_s[1]):int(rect_s[3]), int(rect_s[0]):int(rect_s[2])]
    rect_ori_s_right=laplace_s_ori[:,:,int(rect_s[5]):int(rect_s[7]), int(rect_s[4]):int(rect_s[6])]
    rect_laplace_s_right=laplace_s[:,:,int(rect_s[5]):int(rect_s[7]), int(rect_s[4]):int(rect_s[6])]

    c_ori_left=ops.zeros([data_c.shape[0],data_c.shape[1],tar_w,tar_h],dtype=ms.float32)  # .to("cuda")
    c_mask_left=ops.zeros([data_c.shape[0],data_c.shape[1],tar_w,tar_h],dtype=ms.float32)  # .to("cuda")
    s_ori_left=ops.zeros([data_s.shape[0],data_s.shape[1],tar_w,tar_h],dtype=ms.float32)  # .to("cuda")
    s_laplace_left=ops.zeros([data_s.shape[0],data_s.shape[1],tar_w,tar_h],dtype=ms.float32)  # .to("cuda")
    c_ori_left[:,:,int(tar_w//2-c_w_left//2):int(tar_w//2+c_w_left//2+c_w_left%2),int(tar_h//2-c_h_left//2):int(tar_h//2+c_h_left//2+c_h_left%2)]=rect_ori_c_left
    c_mask_left[:,:,int(tar_w//2-c_w_left//2):int(tar_w//2+c_w_left//2+c_w_left%2),int(tar_h//2-c_h_left//2):int(tar_h//2+c_h_left//2+c_h_left%2)]=rect_mask_c_left

    s_ori_left[:,:,int(tar_w//2-s_w_left//2):int(tar_w//2+s_w_left//2+s_w_left%2),int(tar_h//2-s_h_left//2):int(tar_h//2+s_h_left//2+s_h_left%2)]=rect_ori_s_left

    s_laplace_left[:,:,int(tar_w//2-s_w_left//2):int(tar_w//2+s_w_left//2+s_w_left%2),int(tar_h//2-s_h_left//2):int(tar_h//2+s_h_left//2+s_h_left%2)]=rect_laplace_s_left


    lms_c_left = lms_c[0]-np.array([center_c_left_x, center_c_left_y])
    lms_c_right = lms_c[1]-np.array([center_c_right_x, center_c_right_y])
    lms_s_left = lms_s[0]-np.array([center_s_left_x, center_s_left_y])
    lms_s_right = lms_s[1]-np.array([center_s_right_x, center_s_right_y])

    c_ori_right=ops.zeros([data_c.shape[0],data_c.shape[1],tar_w,tar_h],dtype=ms.float32)  # .to("cuda")
    c_mask_right=ops.zeros([data_c.shape[0],data_c.shape[1],tar_w,tar_h],dtype=ms.float32)  # .to("cuda")
    s_ori_right=ops.zeros([data_s.shape[0],data_s.shape[1],tar_w,tar_h],dtype=ms.float32)  # .to("cuda")
    s_laplace_right=ops.zeros([data_s.shape[0],data_s.shape[1],tar_w,tar_h],dtype=ms.float32)  # .to("cuda")
    c_ori_right[:,:,int(tar_w//2-c_w_right//2):int(tar_w//2+c_w_right//2+c_w_right%2),int(tar_h//2-c_h_right//2):int(tar_h//2+c_h_right//2+c_h_right%2)]=rect_ori_c_right
    c_mask_right[:,:,int(tar_w//2-c_w_right//2):int(tar_w//2+c_w_right//2+c_w_right%2),int(tar_h//2-c_h_right//2):int(tar_h//2+c_h_right//2+c_h_right%2)]=rect_mask_c_right

    s_ori_right[:,:,int(tar_w//2-s_w_right//2):int(tar_w//2+s_w_right//2+s_w_right%2),int(tar_h//2-s_h_right//2):int(tar_h//2+s_h_right//2+s_h_right%2)]=rect_ori_s_right

    s_laplace_right[:,:,int(tar_w//2-s_w_right//2):int(tar_w//2+s_w_right//2+s_w_right%2),int(tar_h//2-s_h_right//2):int(tar_h//2+s_h_right//2+s_h_right%2)]=rect_laplace_s_right

    if flag:
        mark = judge_local(rect_maskball_s_left, rect_maskball_s_right, lms_s)
    else:
        mark = judge_local_brow(rect_ori_s_left, rect_ori_s_right, lms_s)
    if mark == True:
        s_ori_right=ops.zeros([data_s.shape[0],data_s.shape[1],tar_w,tar_h],dtype=ms.float32)  # .to("cuda")
        s_laplace_right=ops.zeros([data_s.shape[0],data_s.shape[1],tar_w,tar_h],dtype=ms.float32)  # .to("cuda")
        for i in range(s_ori_right.shape[3]):
            s_ori_right[:,:,:,i]=s_ori_left[:,:,:,s_ori_right.shape[3]-i-1]
            s_laplace_right[:,:,:,i]=s_laplace_left[:,:,:,s_laplace_right.shape[3]-i-1]
        #if flag:
        lms_s_media = np.vstack((lms_s_left, lms_s_left))
        if flag:
            lms_s_right = lms_s_media[4:10]
        else:
            lms_s_right = lms_s_media[5:14]
        lms_s_right[:,1] = -lms_s_right[:,1]
        lms_s_right = lms_s_right[::-1]

        s_ori_right  = ops.interpolate(s_ori_right,size=(int(c_w_right/s_w_left*tar_w),int(c_h_right/s_h_left*tar_h)))
        lms_s_right[:,0],lms_s_right[:,1] = lms_c_right[:,0]/s_w_left*c_w_right,lms_s_right[:,1]/s_h_left*c_h_right
        s_ori_left  = ops.interpolate(s_ori_left,size=(int(c_w_left/s_w_left*tar_w),int(c_h_left/s_h_left*tar_h)))
        lms_s_left[:,0],lms_s_left[:,1] = lms_c_left[:,0]/s_w_left*c_w_left,lms_s_left[:,1]/s_h_left*c_h_left
    
    if mark == False:
        s_ori_left=ops.zeros([data_s.shape[0],data_s.shape[1],tar_w,tar_h],dtype=ms.float32)  # .to("cuda")
        s_laplace_left=ops.zeros([data_s.shape[0],data_s.shape[1],tar_w,tar_h],dtype=ms.float32)  # .to("cuda")
        for i in range(s_ori_left.shape[3]):
            s_ori_left[:,:,:,i]=s_ori_right[:,:,:,s_ori_left.shape[3]-i-1]
            s_laplace_left[:,:,:,i]=s_laplace_right[:,:,:,s_laplace_left.shape[3]-i-1]
        #if flag:
        lms_s_media = np.vstack((lms_s_right, lms_s_right))
        if flag:
            lms_s_left = lms_s_media[4:10]
        else:
            lms_s_left = lms_s_media[5:14]
        lms_s_left[:,1] = -lms_s_left[:,1]
        lms_s_left = lms_s_left[::-1]

        s_ori_right  = ops.interpolate(s_ori_right,size=(int(c_w_right/s_w_right*tar_w),int(c_h_right/s_h_right*tar_h)))
        lms_s_right[:,0],lms_s_right[:,1] = lms_c_right[:,0]/s_w_right*c_w_right,lms_s_right[:,1]/s_h_right*c_h_right
        s_ori_left  = ops.interpolate(s_ori_left,size=(int(c_w_right/s_w_right*tar_w),int(c_h_right/s_h_right*tar_h)))
        lms_s_left[:,0],lms_s_left[:,1] = lms_c_left[:,0]/s_w_right*c_w_left,lms_s_left[:,1]/s_h_right*c_h_left
    
    if flag==True:
        s_ori_left_line = s_ori_left[:,:,:,int(lms_s_left[3][1]+s_ori_left.shape[3]//2)-4:]
        s_ori_left_line = ops.interpolate(s_ori_left_line,size=(s_ori_left_line.shape[2],int(s_ori_left_line.shape[3]*1.0)))[:,:,:,:s_ori_left_line.shape[3]]
        s_ori_left[:,:,:,int(lms_s_left[3][1]+s_ori_left.shape[3]//2)-4:]= s_ori_left_line
    warp_s_ori_left,warp_s_laplace_left = crop_face_best_fit(c_ori_left,lms_c_left,s_ori_left,lms_s_left,s_laplace_left)
    if flag:
        warp_s_ori_left = warp_s_ori_left*c_mask_left
        warp_s_laplace_left = warp_s_laplace_left*c_mask_left

    if flag==True:
        s_ori_right_line = s_ori_right[:,:,:,:int(lms_s_right[0][1]+s_ori_right.shape[3]//2)+4]
        s_ori_right_line = ops.interpolate(s_ori_right_line,size=(s_ori_right_line.shape[2],int(s_ori_right_line.shape[3]*1.0)))[:,:,:,-int(lms_s_right[0][1]+s_ori_right.shape[3]//2+4):]
        s_ori_right[:,:,:,:int(lms_s_right[0][1]+s_ori_right.shape[3]//2)+4]=s_ori_right_line
    warp_s_ori_right,warp_s_laplace_right = crop_face_best_fit(c_ori_right,lms_c_right,s_ori_right,lms_s_right,s_laplace_right)
    if flag:
        warp_s_ori_right = warp_s_ori_right*c_mask_right
        warp_s_laplace_right = warp_s_laplace_right*c_mask_right

    align_s = ops.zeros_like(laplace_c, dtype=ms.float32)  # .to("cuda")
    align_s_ori = ops.zeros_like(laplace_c_ori, dtype=ms.float32)  # .to("cuda")

    center_c_left_x, center_c_left_y =int((rect_c[1]+rect_c[3])//2), int((rect_c[0]+rect_c[2])//2)
    center_c_right_x, center_c_right_y =int((rect_c[5]+rect_c[7])//2), int((rect_c[4]+rect_c[6])//2)

    align_s[:,:,int(max(0,center_c_left_x-tar_w//2)):int(center_c_left_x+tar_w//2), int(center_c_left_y-tar_h//2):int(center_c_left_y+tar_h//2)]=warp_s_laplace_left[:,:,int(abs(center_c_left_x-tar_w//2-max(0, center_c_left_x-tar_w//2))):,:]
    align_s[:,:,int(max(0,center_c_right_x-tar_w//2)):int(center_c_right_x+tar_w//2), int(max(0,center_c_right_y-tar_h//2)):int(center_c_right_y+tar_h//2)]=warp_s_laplace_right[:,:,int(abs(center_c_right_x-tar_w//2-max(0, center_c_right_x-tar_w//2))):,int(abs(center_c_right_y-tar_h//2-max(0,center_c_right_y-tar_h//2))):]

    align_s_ori[:,:,int(max(0,center_c_left_x-tar_w//2)):int(center_c_left_x+tar_w//2), int(center_c_left_y-tar_h//2):int(center_c_left_y+tar_h//2)]=warp_s_ori_left[:,:,int(abs(center_c_left_x-tar_w//2-max(0, center_c_left_x-tar_w//2))):,:]
    align_s_ori[:,:,int(max(0,center_c_right_x-tar_w//2)):int(center_c_right_x+tar_w//2), int(max(0,center_c_right_y-tar_h//2)):int(center_c_right_y+tar_h//2)]=warp_s_ori_right[:,:,int(abs(center_c_right_x-tar_w//2-max(0,center_c_right_x-tar_w//2))):,int(abs(center_c_right_y-tar_h//2-max(0,center_c_right_y-tar_h//2))):]
    
    return align_s, align_s_ori


# def tensor_mean_std(mask_c_noskin1, c_tnet):
#     eps=1e-5
#     mask_c_noskin1_size = len(torch.nonzero(mask_c_noskin1))
#     mask_c_noskin1_mean = torch.sum(torch.sum(mask_c_noskin1*c_tnet,2,keepdim=True),3,keepdim=True)/(mask_c_noskin1_size+eps)
#     mask_c_noskin1_std = torch.sqrt(torch.sum(torch.sum((mask_c_noskin1*c_tnet-mask_c_noskin1_mean*mask_c_noskin1).pow(2),2,keepdim=True),3,keepdim=True)/(mask_c_noskin1_size+eps)+eps)
#     return mask_c_noskin1_mean,mask_c_noskin1_std
# def pool_mean_std(mask_c_noskin1,c_tnet):
#     m=torch.nn.AvgPool2d(4,stride=4)
#     mask_c_noskin1_mean = m(mask_c_noskin1*c_tnet)
#     mask_c_noskin1_mean=F.interpolate(mask_c_noskin1_mean, size=c_tnet.shape[2:], mode='nearest').repeat(1, 1, 1, 1)
#     mask_c_noskin1_std = torch.sqrt(m((c_tnet-mask_c_noskin1_mean).pow(2)))
#     mask_c_noskin1_std =F.interpolate(mask_c_noskin1_std, size=c_tnet.shape[2:], mode='nearest').repeat(1, 1, 1, 1)
#     return mask_c_noskin1_mean,mask_c_noskin1_std

def get_mean_std(image,mask,total_nums):
    img_mean = np.sum(image * mask)/ total_nums
    var = (image - img_mean) * (image - img_mean) * mask
    var = np.sum(var) / total_nums
    std = math.sqrt(var)
    return img_mean, std
    

def image_stats(image,mask):
    total_nums = np.sum(mask)
    (l, a, b) = cv2.split(image)
    l_mean, l_std = get_mean_std(l, mask, total_nums)
    a_mean, a_std = get_mean_std(a, mask, total_nums)
    b_mean, b_std = get_mean_std(b, mask, total_nums)
    return (l_mean, l_std, a_mean, a_std, b_mean, b_std)

# source 原图；mask_source 原图0-255二值化；destination 目标图； mask_dest 目标图0-255二值化。
# 此函数没有进行alpha-blend处理。
# def color_transfer(data_c, mask_c, data_s,mask_s):
#     source = de_norm(data_c).squeeze(0).permute(1,2,0).cpu().numpy() * 255.0
#     source = cv2.cvtColor(source.astype("uint8"), cv2.COLOR_RGB2LAB).astype("float")
#     destination = de_norm(data_s).squeeze(0).permute(1,2,0).cpu().numpy() * 255.0
#     destination = cv2.cvtColor(destination.astype("uint8"), cv2.COLOR_RGB2LAB).astype("float")
#     mask_source =mask_c.squeeze().cpu().numpy()
#     mask_dest =mask_s.squeeze().cpu().numpy()
    
#     mask_inverse = 1-mask_dest
#     (l_mean_src, l_std_src, a_mean_src, a_std_src, b_mean_src, b_std_src) = image_stats(source, mask_source)
#     (l_mean_dest, l_std_dest, a_mean_dest, a_std_dest, b_mean_dest, b_std_dest) = image_stats(destination, mask_dest)
#     (l_mean_inv, l_std_inv, a_mean_inv, a_std_inv, b_mean_inv, b_std_inv) = image_stats(destination, mask_inverse)
#     (l, a, b) = cv2.split(destination)

#     l_mean_dest = mask_dest * l_mean_dest
#     a_mean_dest = mask_dest * a_mean_dest
#     b_mean_dest = mask_dest * b_mean_dest

#     l -= l_mean_dest
#     a -= a_mean_dest
#     b -= b_mean_dest

#     l -= l_mean_inv*mask_inverse
#     a -= a_mean_inv*mask_inverse
#     b -= b_mean_inv*mask_inverse

#     l += l_mean_src
#     a += a_mean_src
#     b += b_mean_src

#     l = np.clip(l, 0, 255)
#     a = np.clip(a, 0, 255)
#     b = np.clip(b, 0, 255)

#     transfer = cv2.merge([l, a, b])
#     transfer = cv2.cvtColor(transfer.astype("uint8"), cv2.COLOR_LAB2BGR)
#     transfer = cv2.cvtColor(transfer,cv2.COLOR_BGR2RGB)
#     res = torch.tensor(transfer, dtype=torch.float).to("cuda").unsqueeze(0).permute(0,3,1,2)
#     res = (res/255-0.5)*2

#     return res


def spp_mean_std_for_adain(x, window):
    # import pdb;pdb.set_trace()
    eps=1e-5
    _, _, h, w = x.shape
    kernel_size = (math.ceil(h / window), math.ceil(w / window))
    stride = (math.ceil(h / window), math.ceil(w / window))
    pooling = (math.floor((kernel_size[0]*window-h+1)/2), math.floor((kernel_size[1]*window-w+1)/2))
    x_pad = ops.pad(x, [pooling[1],pooling[1],pooling[0],pooling[0]])
    mean = ops.avg_pool2d(x_pad, kernel_size=kernel_size, stride=stride)
    mean_repeat=ops.interpolate(mean, size=x.shape[2:], mode='nearest').tile((1, 1, 1, 1))

    square = (x - mean_repeat).pow(2)
    square = ops.pad(square, [pooling[1],pooling[1],pooling[0],pooling[0]])
    var = ops.avg_pool2d(square, kernel_size=kernel_size, stride=stride) #sum
    std = ops.sqrt(var+eps)
    return mean, mean_repeat, std


def spp_adain(mask_c, rect_c, c, rect_s, s, window, i):
    eps=1e-5
    if rect_c.shape[0] == 1:
        newrect_c = (rect_c.squeeze(0) / int(i)).int()
    else:
        newrect_c = (rect_c / int(i)).int()
    if rect_s.shape[0] == 1:
        newrect_s = (rect_s.squeeze(0) / int(i)).int()
    else:
        newrect_s = (rect_s / int(i)).int()
    # import pdb;pdb.set_trace()
    new_c_lefteye  = c[:, :, newrect_c[5]:newrect_c[7], newrect_c[4]:newrect_c[6]]
    new_c_righteye = c[:, :, newrect_c[1]:newrect_c[3], newrect_c[0]:newrect_c[2]]

    new_s_lefteye  = s[:, :, newrect_s[5]:newrect_s[7], newrect_s[4]:newrect_s[6]]
    new_s_righteye = s[:, :, newrect_s[1]:newrect_s[3], newrect_s[0]:newrect_s[2]]

    #left_eye_c
    mean_lefteye_c, mean_lefteye_c_up, std_lefteye_c = spp_mean_std_for_adain(new_c_lefteye, window)
    std_lefteye_c_up=ops.interpolate(std_lefteye_c, size=new_c_lefteye.shape[2:], mode='nearest').tile((1, 1, 1, 1))

    #right_eye_c
    mean_righteye_c, mean_righteye_c_up, std_righteye_c = spp_mean_std_for_adain(new_c_righteye, window)
    std_righteye_c_up=ops.interpolate(std_righteye_c, size=new_c_righteye.shape[2:], mode='nearest').tile((1, 1, 1, 1))


    #left_eye_s
    mean_lefteye_s, mean_lefteye_s_repeat, std_lefteye_s = spp_mean_std_for_adain(new_s_lefteye, window)
    mean_lefteye_s_up=ops.interpolate(mean_lefteye_s, size=new_c_lefteye.shape[2:], mode='nearest').tile((1, 1, 1, 1))
    std_lefteye_s_up=ops.interpolate(std_lefteye_s, size=new_c_lefteye.shape[2:], mode='nearest').tile((1, 1, 1, 1))

    #right_eye_s
    mean_righteye_s, mean_righteye_s_repeat, std_righteye_s = spp_mean_std_for_adain(new_s_righteye, window)
    mean_righteye_s_up=ops.interpolate(mean_righteye_s, size=new_c_righteye.shape[2:], mode='nearest').tile((1, 1, 1, 1))
    std_righteye_s_up=ops.interpolate(std_righteye_s, size=new_c_righteye.shape[2:], mode='nearest').tile((1, 1, 1, 1))

    # import pdb;pdb.set_trace()
    new_c_lefteye = ((new_c_lefteye-mean_lefteye_c_up))/(std_lefteye_c_up+eps)*std_lefteye_s_up+mean_lefteye_s_up
    new_c_righteye = ((new_c_righteye-mean_righteye_c_up))/(std_righteye_c_up+eps)*std_righteye_s_up+mean_righteye_s_up

    #TODO clone or copy 
    new_c = c.copy()
    new_c[:, :, newrect_c[5]:newrect_c[7], newrect_c[4]:newrect_c[6]] = new_c_lefteye
    new_c[:, :, newrect_c[1]:newrect_c[3], newrect_c[0]:newrect_c[2]] = new_c_righteye

    new_c = new_c * mask_c + c * (1 - mask_c)

    return new_c


def spp_mean_std(mask, c, window):
    # import pdb;pdb.set_trace()
    eps=1e-5
    mask_c = mask * c
    mask_ = ops.avg_pool2d(mask * window * window, kernel_size=(window, window), stride=window, padding=0) #sum

    mean = ops.avg_pool2d(mask_c * window * window, kernel_size=(window, window), stride=window, padding=0)
    mean = mean / (mask_+eps)
    mean_repeat = mean.repeat_interleave(window, dim=2).repeat_interleave(window, dim=3)
    assert mean_repeat.shape[2:] == mask_c.shape[2:]
    square = (mask_c - mean_repeat * mask).pow(2)
    var = ops.avg_pool2d(square * window * window, kernel_size=(window, window), stride=window, padding=0) #sum
    
    std = ops.sqrt(var/(mask_+eps)+eps)
    std_repeat = std.repeat_interleave(window, dim=2).repeat_interleave(window, dim=3)

    return mean_repeat, std_repeat

def spp_adain_old(mask_c, c, mask_s, s, window):
    # import pdb;pdb.set_trace()
    eps=1e-5
    mean_c, std_c = spp_mean_std(mask_c, c, window)
    mean_s, std_s = spp_mean_std(mask_s, s, window)

    new_c_tnet = ((mask_c*c-mean_c*mask_c))/(std_c*mask_c+eps)*std_s*mask_c+mean_s*mask_c+(1-mask_c)*c

    return new_c_tnet


class MultiBranch(nn.Cell):
    def __init__(self, in_channels, max_win) -> None:
        super(MultiBranch, self).__init__()
        self.control_v1 = nn.Conv2d(in_channels, 1, (1, 1), pad_mode='valid', has_bias=True)
        self.relu_v1 = nn.Sigmoid()
        self.max_win = max_win
        
    def construct(self, feat_c, mask_c, c_tnet, mask_s, s):
        if ops.sum(mask_s)>0 and ops.sum(mask_c)>0:
            feat = self.relu_v1(self.control_v1(feat_c))
            c_tnet_64 = spp_adain_old(mask_c=mask_c, c=c_tnet, mask_s=mask_s, s=s, window=self.max_win)
            c_tnet_1 = spp_adain_old(mask_c=mask_c, c=c_tnet, mask_s=mask_s, s=s, window=1)
            new_c_tnet = feat * c_tnet_64 + (1. - feat) * c_tnet_1
            return new_c_tnet
        else:
            return c_tnet


class MultiBranchEye(nn.Cell):
    def __init__(self, in_channels, max_win, downsample_factor) -> None:
        super(MultiBranchEye, self).__init__()
        self.control_v1 = nn.Conv2d(in_channels, 3, (1, 1), pad_mode='valid', has_bias=True)
        self.lr = nn.LeakyReLU(0.01)
        self.relu_v1 = nn.Softmax(axis=1)
        self.max_win = max_win
        self.downsample_factor = downsample_factor
        
    def construct(self, feat_c, mask_c, rect_c, c_tnet, mask_s, rect_s, s):
        feat = self.relu_v1(self.lr(self.control_v1(feat_c)))

        c_tnet_6 = spp_adain(mask_c, rect_c, c_tnet, rect_s, s, window=6, i=self.downsample_factor)

        if ops.sum(mask_s)>0 and ops.sum(mask_c)>0:
            c_tnet_64 = spp_adain_old(mask_c=mask_c, c=c_tnet, mask_s= mask_s, s=s, window=self.max_win)
            c_tnet_1 = spp_adain_old(mask_c=mask_c, c=c_tnet, mask_s=mask_s, s=s, window=1)
            new_c_tnet = feat[:, [0], :, :] * c_tnet_64 + feat[:, [1], :, :] * c_tnet_6 + feat[:, [2], :, :] * c_tnet_1
        else:
            new_c_tnet = c_tnet_6
            
        return new_c_tnet


class Generator(nn.Cell, Track):
    """Generator. Encoder-Decoder Architecture."""
    def __init__(self, config):
        super(Generator, self).__init__()

        # -------------------------- PNet(MDNet) for obtaining makeup matrices --------------------------

        layers = nn.SequentialCell(
            nn.Conv2d(3, 64, kernel_size=7, stride=1, pad_mode='pad', padding=3, has_bias=False),
            nn.InstanceNorm2d(64, affine=True),
            nn.ReLU()
        )
        self.pnet_in = layers

        # Down-Sampling
        curr_dim = 64
        for i in range(2):
            layers = nn.SequentialCell(
                nn.Conv2d(curr_dim, curr_dim * 2, kernel_size=4, stride=2, pad_mode='pad', padding=1, has_bias=False),
                nn.InstanceNorm2d(curr_dim * 2, affine=True),
                nn.ReLU(),
            )

            setattr(self, f'pnet_down_{i+1}', layers)
            curr_dim = curr_dim * 2

        for i in range(3):
            setattr(self, f'pnet_bottleneck_{i+1}', ResidualBlock(dim_in=curr_dim, dim_out=curr_dim, net_mode='p'))

        # --------------------------- TNet(MANet) for applying makeup transfer ----------------------------

        self.tnet_in_conv = nn.Conv2d(3, 64, kernel_size=7, stride=1, pad_mode='pad', padding=3, has_bias=False)
        self.tnet_in_spade = nn.InstanceNorm2d(64, affine=True)
        self.tnet_in_relu = nn.ReLU()

        # Down-Sampling
        curr_dim = 64

        for i in range(2):
            setattr(self, f'tnet_down_conv_{i+1}', nn.Conv2d(curr_dim, curr_dim * 2, kernel_size=4, stride=2, pad_mode='pad', padding=1, has_bias=False))
            setattr(self, f'tnet_down_spade_{i+1}', nn.InstanceNorm2d(curr_dim * 2, affine=True))
            setattr(self, f'tnet_down_relu_{i+1}', nn.ReLU())
            curr_dim = curr_dim * 2

        # Bottleneck
        
        for i in range(6):
            setattr(self, f'tnet_bottleneck_{i+1}', ResidualBlock(dim_in=curr_dim, dim_out=curr_dim, net_mode='p'))
        
        # Up-Sampling
        for i in range(2):
            setattr(self, f'tnet_up_conv_{i+1}', nn.Conv2dTranspose(curr_dim, curr_dim // 2, kernel_size=4, stride=2, pad_mode='pad', padding=1, has_bias=False))
            setattr(self, f'tnet_up_spade_{i+1}', nn.InstanceNorm2d(curr_dim // 2, affine=True))
            setattr(self, f'tnet_up_relu_{i+1}', nn.ReLU())
            curr_dim = curr_dim // 2

        layers = nn.SequentialCell(
            nn.Conv2d(curr_dim, 3, kernel_size=7, stride=1, pad_mode='pad', padding=3, has_bias=False),
            nn.Tanh()
        )
        self.tnet_out = layers

        self.conv1 = nn.Conv2d(128, 256, kernel_size=1, stride=1, pad_mode='pad', padding=0, has_bias=True)
        self.conv2 = nn.Conv2d(128, 256, kernel_size=1, stride=1, pad_mode='pad', padding=0, has_bias=True)

        self.mean_std_windows = config.TRAINING.WINDOWS
        print('mean_std_windows in net spp encoder multi attention all nonzero unify --> ', self.mean_std_windows)
        self.multibranch_eye = nn.CellList([MultiBranchEye(128 * 2, max_win=128, downsample_factor=2), MultiBranchEye(256 * 2, max_win=64, downsample_factor=4)])
        self.multibranch_noskin = nn.CellList([MultiBranch(128 * 2, max_win=128), MultiBranch(256 * 2, max_win=64)])
        self.multibranch_skin = MultiBranch(512, max_win=64)

        Track.__init__(self)

    def construct(self, c, s, mask_c, mask_s, diff_c, diff_s, rect_c, rect_s, lms_c, lms_s, gamma=None, beta=None, ret=False,epoch = None, warp=True):
        if lms_c.shape[0] != 1:
            lms_c = lms_c.asnumpy()
        else:
            lms_c = lms_c.squeeze(0).asnumpy()
        if lms_s.shape[0] != 1:
            lms_s = lms_s.asnumpy()
        else:
            lms_s = lms_s.squeeze(0).asnumpy()
        c, s, mask_c, mask_s, diff_c, diff_s = [x.squeeze(0) if x.ndim == 5 else x for x in [c, s, mask_c, mask_s, diff_c, diff_s]]
        """attention version
        c: content, stands for source image. shape: (b, c, h, w)
        s: style, stands for reference image. shape: (b, c, h, w)
        mask_list_c: lip, skin, eye. (b, 1, h, w)
        """

        if warp:
            warp_eye_laplace, warp_eye_ori = laplace_target_wrap(c, s, (mask_c[2]+mask_c[3]).unsqueeze(0), [mask_s[2],mask_s[5],mask_s[3],mask_s[1]+mask_s[6]], rect_c, rect_s, \
                    lms_c=np.array([lms_c[58:64], lms_c[52:58]]), lms_s=np.array([lms_s[58:64], lms_s[52:58]]), flag=True)
        else:
            warp_eye_laplace, warp_eye_ori = -1, -1

        self.track("start")
        middle_pre = []
        c_tnet = self.tnet_in_conv(c)
        s = self.pnet_in(s)
        middle_pre.append(s)
        c_tnet = self.tnet_in_spade(c_tnet)
        c_tnet = self.tnet_in_relu(c_tnet)

        # down-sampling
        for i in range(2):
            if gamma is None:
                cur_pnet_down = getattr(self, f'pnet_down_{i+1}')
                s = cur_pnet_down(s)
                middle_pre.append(s)

            cur_tnet_down_conv = getattr(self, f'tnet_down_conv_{i+1}')
            cur_tnet_down_spade = getattr(self, f'tnet_down_spade_{i+1}')
            cur_tnet_down_relu = getattr(self, f'tnet_down_relu_{i+1}')

            c_tnet = cur_tnet_down_conv(c_tnet)
            c_tnet = cur_tnet_down_spade(c_tnet)
            c_tnet = cur_tnet_down_relu(c_tnet)
            # import pdb;pdb.set_trace()
            eps=1e-5
            #eye adain
            mask_c_eye=ops.interpolate(mask_c[2].unsqueeze(0), size=c_tnet.shape[2:], mode='nearest').tile((1, 1, 1, 1))
            mask_s_eye=ops.interpolate(mask_s[2].unsqueeze(0), size=c_tnet.shape[2:], mode='nearest').tile((1, 1, 1, 1))

            c_tnet = self.multibranch_eye[i](ops.cat([c_tnet, s], axis=1), mask_c_eye, rect_c, c_tnet, mask_s_eye, rect_s, s)

            #lip adain
            mask_c_noskin = mask_c[0,:,:,:].unsqueeze(0)
            mask_s_noskin = mask_s[0,:,:,:].unsqueeze(0)

            mask_c_noskin1=ops.interpolate(mask_c_noskin, size=c_tnet.shape[2:], mode='nearest').tile((1, 1, 1, 1))
            mask_s_noskin1=ops.interpolate(mask_s_noskin, size=c_tnet.shape[2:], mode='nearest').tile((1, 1, 1, 1))

            c_tnet = self.multibranch_noskin[i](ops.cat([c_tnet, s], axis=1), mask_c_noskin1, c_tnet, mask_s_noskin1, s)


        self.track("downsampling")

        # bottleneck
        middle_s = []
        for i in range(6):
            if gamma is None and i <= 2:
                cur_pnet_bottleneck = getattr(self, f'pnet_bottleneck_{i+1}')
            cur_tnet_bottleneck = getattr(self, f'tnet_bottleneck_{i+1}')

            # get s_pnet from p and transform
            if i == 3:
                #skin_mask
                mask_skin_c = mask_c[1,:,:,:].unsqueeze(0) + mask_c[4,:,:,:].unsqueeze(0) 
                mask_skin_s = mask_s[1,:,:,:].unsqueeze(0) + mask_s[4,:,:,:].unsqueeze(0)

                mask_skin_c1=ops.interpolate(mask_skin_c, size=c_tnet.shape[2:], mode='nearest').tile((1, 1, 1, 1))
                mask_skin_s1=ops.interpolate(mask_skin_s, size=s.shape[2:], mode='nearest').tile((1, 1, 1, 1))
                    
                mask_all = ops.zeros_like(mask_skin_c1)+1

                c_tnet = self.multibranch_skin(ops.cat([c_tnet, s], axis=1), mask_all, c_tnet, mask_skin_s1, s)

            if gamma is None and i <= 2:
                s = cur_pnet_bottleneck(s)
                middle_s.append(s)
            c_tnet = cur_tnet_bottleneck(c_tnet)

        self.track("bottleneck")
        # up-sampling
        for i in range(2):
            cur_tnet_up_conv = getattr(self, f'tnet_up_conv_{i+1}')
            cur_tnet_up_spade = getattr(self, f'tnet_up_spade_{i+1}')
            cur_tnet_up_relu = getattr(self, f'tnet_up_relu_{i+1}')

            c_tnet = cur_tnet_up_conv(c_tnet)
            c_tnet = cur_tnet_up_spade(c_tnet)
            c_tnet = cur_tnet_up_relu(c_tnet)

        self.track("upsampling")

        c_tnet = self.tnet_out(c_tnet)
        return c_tnet,  [warp_eye_ori, warp_eye_laplace]


class Discriminator(nn.Cell):
    """Discriminator. PatchGAN."""
    def __init__(self, image_size=128, conv_dim=64, repeat_num=3, norm='SN'):
        super(Discriminator, self).__init__()

        layers = []
        if norm=='SN':
            layers.append(SpectralNorm(nn.Conv2d(3, conv_dim, kernel_size=4, stride=2, pad_mode='pad', padding=1, has_bias=True)))
        else:
            layers.append(nn.Conv2d(3, conv_dim, kernel_size=4, stride=2, pad_mode='pad', padding=1, has_bias=True))
        layers.append(nn.LeakyReLU(0.01))

        curr_dim = conv_dim
        for i in range(1, repeat_num):
            if norm=='SN':
                layers.append(SpectralNorm(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, pad_mode='pad', padding=1, has_bias=True)))
            else:
                layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, pad_mode='pad', padding=1, has_bias=True))
            layers.append(nn.LeakyReLU(0.01))
            curr_dim = curr_dim * 2

        if norm=='SN':
            layers.append(SpectralNorm(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=1, pad_mode='pad', padding=1, has_bias=True)))
        else:
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=1, pad_mode='pad', padding=1, has_bias=True))
        layers.append(nn.LeakyReLU(0.01))
        curr_dim = curr_dim *2

        self.main = nn.SequentialCell(*layers)
        if norm=='SN':
            self.conv1 = SpectralNorm(nn.Conv2d(curr_dim, 1, kernel_size=4, stride=1, pad_mode='pad', padding=1, has_bias=False))
        else:
            self.conv1 = nn.Conv2d(curr_dim, 1, kernel_size=4, stride=1, pad_mode='pad', padding=1, has_bias=False)


    def construct(self, x):
        if x.ndim == 5:
            x = x.squeeze(0)
        assert x.ndim == 4, x.ndim
        h = self.main(x)
        out_makeup = self.conv1(h)
        return out_makeup



# def minmaxscaler(data):
#     data_res = torch.zeros_like(data, dtype=torch.float)
#     max_value=torch.max(torch.max(data,2, keepdim=True)[0],3, keepdim=True)[0]
#     min_value=torch.min(torch.min(data,2, keepdim=True)[0],3, keepdim=True)[0]
#     data_res = (data-min_value)/(max_value-min_value)

#     return data_res

# def strong_eye(data):
#     laplace_input_abs=torch.abs(data)
#     mean_value=torch.mean(torch.mean(laplace_input_abs,dim=2,keepdim=True),dim=3,keepdim=True)
#     laplace_input_trans=1.0/(1+torch.exp(-(laplace_input_abs-mean_value)))
#     return laplace_input_trans





class Discriminatoreye(nn.Cell):
    def __init__(self, image_size=120, conv_dim=64, repeat_num=2, norm='SN'):
        super(Discriminatoreye, self).__init__()

        layers=[]
        if norm=='SN':
            layers.append(SpectralNorm(nn.Conv2d(3, conv_dim, kernel_size=3, stride=2, padding=1)))
        else:
            layers.append(nn.Conv2d(3, conv_dim, kernel_size=3, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.01, inplace=True))
        curr_dim = conv_dim
        for i in range(1, repeat_num):
            if norm=='SN':
                layers.append(SpectralNorm(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=3, stride=2, padding=1)))
            else:
                layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1))
            layers.append(nn.LeakyReLU(0.01, inplace=True))
            curr_dim=curr_dim*2

        if norm=='SN':
            self.conv1 = SpectralNorm(nn.Conv2d(curr_dim, 1, kernel_size=3, stride=1, padding=1, bias=False))
        else:
            self.conv1 = nn.Conv2d(curr_dim, 1, kernel_size=3, stride=1, padding=1, bias=False)


        self.main = nn.Sequential(*layers)

    # def construct(self,x):
    #     img=x[0]
    #     mask=x[1]
    #     rect=x[2]
    #     flag = x[3]
    #     if not flag:
    #         kernel = torch.tensor(np.broadcast_to(np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]]), (img.size(0), img.size(1), 3, 3)),dtype=torch.float32).to("cuda")
    #         x=F.conv2d(img, kernel, bias=None, stride=1, padding=1, dilation=1, groups=1)
    #         x=x*mask
    #     else:
    #         x=img*mask
    #     x=x[:,:,int(rect[1]):int(rect[3]), int(rect[0]):int(rect[2])]
    #     w=int(rect[3])-int(rect[1])
    #     h=int(rect[2])-int(rect[0])
    #     input_data=torch.zeros([img.size(0),img.size(1),120,120],dtype=torch.float).to("cuda")
    #     input_data[:,:,60-w//2:60+w//2+w%2,60-h//2:60+h//2+h%2]=x
        
    #     h=self.main(input_data)
    #     out = self.conv1(h)

    #     return h, out

class VGG(nn.Cell):
    def __init__(self, pool='max'):
        super(VGG, self).__init__()
        # vgg modules
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        if pool == 'max':
            self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        elif pool == 'avg':
            self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool3 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool4 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool5 = nn.AvgPool2d(kernel_size=2, stride=2)

    # def construct(self, x, out_keys):
    #     out = {}
    #     out['r11'] = F.relu(self.conv1_1(x))
    #     out['r12'] = F.relu(self.conv1_2(out['r11']))
    #     out['p1'] = self.pool1(out['r12'])
    #     out['r21'] = F.relu(self.conv2_1(out['p1']))
    #     out['r22'] = F.relu(self.conv2_2(out['r21']))
    #     out['p2'] = self.pool2(out['r22'])
    #     out['r31'] = F.relu(self.conv3_1(out['p2']))
    #     out['r32'] = F.relu(self.conv3_2(out['r31']))
    #     out['r33'] = F.relu(self.conv3_3(out['r32']))
    #     out['r34'] = F.relu(self.conv3_4(out['r33']))
    #     out['p3'] = self.pool3(out['r34'])
    #     out['r41'] = F.relu(self.conv4_1(out['p3']))
        
    #     out['r42'] = F.relu(self.conv4_2(out['r41']))
    #     out['r43'] = F.relu(self.conv4_3(out['r42']))
    #     out['r44'] = F.relu(self.conv4_4(out['r43']))
    #     out['p4'] = self.pool4(out['r44'])
    #     out['r51'] = F.relu(self.conv5_1(out['p4']))
    #     out['r52'] = F.relu(self.conv5_2(out['r51']))
    #     out['r53'] = F.relu(self.conv5_3(out['r52']))
    #     out['r54'] = F.relu(self.conv5_4(out['r53']))
    #     out['p5'] = self.pool5(out['r54'])
        
    #     return [out[key] for key in out_keys]


class VGG(TVGG):
    def construct(self, x):
        x = self.features(x)
        return x

def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


# def _vgg(arch, cfg, batch_norm, pretrained, progress, **kwargs):
#     if pretrained:
#         kwargs['init_weights'] = False
#     model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm), **kwargs)
#     if pretrained:
#         model.load_state_dict(torch.load("vgg16-397923af.pth"))
#     return model


# def vgg16(pretrained=False, progress=True, **kwargs):
#     r"""VGG 16-layer model (configuration "D")
#     `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#         progress (bool): If True, displays a progress bar of the download to stderr
#     """
#     return _vgg('vgg16', 'D', False, pretrained, progress, **kwargs)
