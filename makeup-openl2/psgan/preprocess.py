#!/usr/bin/python
# -*- encoding: utf-8 -*-
import os.path as osp
pwd = osp.split(osp.realpath(__file__))[0]
import sys
sys.path.append(pwd + '/..')

import cv2
import numpy as np
from PIL import Image

import mindspore as ms
from mindspore import ops
import mindspore.dataset.transforms as transforms
import mindspore.dataset.vision as vision


import faceutils as futils
# from scripts.utils.face_sdk import ExtractFrame, Landmarks
import face_recognition


transform = transforms.Compose([
    vision.ToTensor(),
    vision.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5],is_hwc=False)])


# def ToTensor(pic):
#     # handle PIL Image
#     if pic.mode == 'I':
#         img = torch.from_numpy(np.array(pic, np.int32, copy=False))
#     elif pic.mode == 'I;16':
#         img = torch.from_numpy(np.array(pic, np.int16, copy=False))
#     else:
#         img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
#     # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
#     if pic.mode == 'YCbCr':
#         nchannel = 3
#     elif pic.mode == 'I;16':
#         nchannel = 1
#     else:
#         nchannel = len(pic.mode)
#     img = img.view(pic.size[1], pic.size[0], nchannel)
#     # put it from HWC to CHW format
#     # yikes, this transpose takes 80% of the loading time/CPU
#     img = img.transpose(0, 1).transpose(0, 2).contiguous()
#     if isinstance(img, torch.ByteTensor):
#         return img.float()
#     else:
#         return img


#TODO Variable
def to_var(x, requires_grad=True):
    if requires_grad:
        # return Variable(x).float()
        return ms.Tensor(x).float()
    else:
        # return Variable(x, requires_grad=requires_grad).float()
        return ms.Tensor(x).float()


def copy_area_list_left(lms):
    rect = [int(min(lms[:, 1])) - PreProcess.eye_margin,
    int(min(lms[:, 0])) - PreProcess.eye_margin-8-4,
    int(max(lms[:, 1])) + PreProcess.eye_margin + 1+16,
    int(max(lms[:, 0])) + PreProcess.eye_margin + 1+4+16]
    return rect

def copy_area_list_right(lms):
    rect = [int(min(lms[:, 1])) - PreProcess.eye_margin-16,
    int(min(lms[:, 0])) - PreProcess.eye_margin-8-4,
    int(max(lms[:, 1])) + PreProcess.eye_margin + 1,
    int(max(lms[:, 0])) + PreProcess.eye_margin + 1+4+16]
    return rect


def copy_area_left(tar, src, lms):
    rect = [int(min(lms[:, 1])) - PreProcess.eye_margin, 
            int(min(lms[:, 0])) - PreProcess.eye_margin-8, 
            int(max(lms[:, 1])) + PreProcess.eye_margin + 1+16, 
            int(max(lms[:, 0])) + PreProcess.eye_margin + 1+4]
    tar[:, :, rect[1]:rect[3], rect[0]:rect[2]] = \
        src[:, :, rect[1]:rect[3], rect[0]:rect[2]]
    # src[:, :, rect[1]+4:rect[3]-8, rect[0]:rect[2]] = 0
    
    rect[3] = rect[3]+8
    rect[1] = rect[1]-4
    # rect[3] = rect[3]+24
    # rect[1] = rect[1]-8
    # rect[2] = rect[2]+8
    tar_cv = np.zeros((src.shape[2],src.shape[3]),dtype = np.uint8)
    center_x,center_y = (rect[1]+rect[3])//2,(rect[0]+rect[2])//2
    size_x,size_y = (rect[3]-rect[1])//2,(rect[2]-rect[0])//2
    tar_cv = cv2.ellipse(tar_cv,(center_y,center_x),(size_x,size_y),0,0,360,(1),-1)
    target = ms.Tensor(tar_cv,dtype=ms.float32).unsqueeze(0).unsqueeze(0)
    src[:, :, rect[1]+4:rect[3]-8, rect[0]:rect[2]] = 0
    return target

def copy_area_right(tar, src, lms):
    rect = [int(min(lms[:, 1])) - PreProcess.eye_margin-16, 
            int(min(lms[:, 0])) - PreProcess.eye_margin-8, 
            int(max(lms[:, 1])) + PreProcess.eye_margin + 1, 
            int(max(lms[:, 0])) + PreProcess.eye_margin + 1+4]
    tar[:, :, rect[1]:rect[3], rect[0]:rect[2]] = \
        src[:, :, rect[1]:rect[3], rect[0]:rect[2]]
    # src[:, :, rect[1]:rect[3], rect[0]:rect[2]] = 0
    rect[3] = rect[3]+8
    rect[1] = rect[1]-4
    # rect[3] = rect[3]+24
    # rect[1] = rect[1]-8
    # rect[0] = rect[0]-8
    tar_cv = np.zeros((src.shape[2],src.shape[3]),dtype = np.uint8)
    center_x,center_y = (rect[1]+rect[3])//2,(rect[0]+rect[2])//2
    size_x,size_y = (rect[3]-rect[1])//2,(rect[2]-rect[0])//2
    tar_cv = cv2.ellipse(tar_cv,(center_y,center_x),(size_x,size_y),0,0,360,(1),-1)
    target = ms.Tensor(tar_cv).unsqueeze(0).unsqueeze(0)
    src[:, :, rect[1]+4:rect[3]-8, rect[0]:rect[2]] = 0
    return target


class PreProcess:
    eye_margin = 16
    diff_size = (64, 64)

    def __init__(self, config, device="cpu", need_parser=True):
        self.device = device
        self.img_size    = config.DATA.IMG_SIZE

        xs, ys = np.meshgrid(
            np.linspace(
                0, self.img_size - 1,
                self.img_size
            ),
            np.linspace(
                0, self.img_size - 1,
                self.img_size
            )
        )
        # xs = xs[None].repeat(config.PREPROCESS.LANDMARK_POINTS, axis=0)
        # ys = ys[None].repeat(config.PREPROCESS.LANDMARK_POINTS, axis=0)
        xs = xs[None].repeat(72, axis=0)
        ys = ys[None].repeat(72, axis=0)
        fix = np.concatenate([ys, xs], axis=0)
        self.fix = ms.Tensor(fix)
        if need_parser:
            self.face_parse = futils.mask.FaceParser(device=device)
        self.up_ratio    = config.PREPROCESS.UP_RATIO
        self.down_ratio  = config.PREPROCESS.DOWN_RATIO
        self.width_ratio = config.PREPROCESS.WIDTH_RATIO
        self.lip_class   = config.PREPROCESS.LIP_CLASS
        self.face_class  = config.PREPROCESS.FACE_CLASS

    def relative2absolute(self, lms):
        return lms * self.img_size

    def process(self, mask, lms, device="cpu"):
        diff = to_var(
            (self.fix.to(ms.float64) - ms.Tensor(lms.transpose((1, 0)
                ).reshape(-1, 1, 1))  # .to(self.device)
            ).unsqueeze(0), requires_grad=False)  # .to(self.device)

        lms_eye_left = lms[58:64]
        lms_eye_right = lms[52:58]
        lms_clone = lms
        lms_clone=to_var(ms.Tensor(lms_clone),requires_grad=False)
        lms = lms.transpose((1, 0)).reshape(-1, 1, 1)   # transpose to (y-x)
        # lms = np.tile(lms, (1, 256, 256))  # (136, h, w)
        diff = to_var((self.fix.to(ms.float64) - ms.Tensor(lms)).unsqueeze(0), requires_grad=False)  # .to(self.device)

        mask_lip = (mask == self.lip_class[0]).float() + (mask == self.lip_class[1]).float()
        mask_face = (mask == self.face_class[0]).float()
        mask_eyebrow = (mask == self.face_class[2]).float()+(mask == self.face_class[3]).float()
        mask_nose = (mask == self.face_class[1]).float()
        mask_eyeball_all = (mask == self.face_class[4]).float()+(mask == self.face_class[5]).float() 


        mask_eyes_ori = ops.zeros_like(mask)
        #TODO clone
        # mask_face_ori = mask_face.clone()
        mask_face_ori = mask_face.copy()
        mask_eyes_left  = copy_area_left(mask_eyes_ori, mask_face, lms_eye_left)
        mask_eyes_right  =copy_area_right(mask_eyes_ori, mask_face, lms_eye_right)
        mask_eyes = mask_eyes_left+mask_eyes_right
        mask_eyes = to_var(mask_eyes, requires_grad=False)*mask_face_ori
        mask_eyes_ori = to_var(mask_eyes_ori, requires_grad=False)

        mask_list = [mask_lip, mask_face, mask_eyes, mask_eyebrow, mask_nose, mask_eyeball_all,mask_eyes_ori]
        mask_aug = ops.cat(mask_list, 0)      # (5, 1, h, w)
        mask_aug_skin = ops.cat([mask_lip],0)
        mask_re = ops.interpolate(mask_aug_skin, size=self.diff_size).tile((1, diff.shape[1], 1, 1))  # (3, 136, 64, 64)
        diff_re = ops.interpolate(diff, size=self.diff_size).tile((1, 1, 1, 1))  # (3, 136, 64, 64)
        diff_re = diff_re * mask_re            # (3, 136, 32, 32)
        norm = ops.norm(diff_re, dim=1, keepdim=True).tile((1, diff_re.shape[1], 1, 1))
        norm = ops.where(norm == 0, ms.Tensor(1e10), norm)
        diff_re /= norm


        rect_left=copy_area_list_left(lms_eye_left)
        rect_right=copy_area_list_right(lms_eye_right)
        rect=ms.Tensor(rect_left+rect_right)
        rect=to_var(rect.int(),requires_grad=False)

        return mask_aug, diff_re, rect, lms_clone

    def __call__(self, image: Image):
        #face = futils.dlib.detect(image)

        #if not face:
        #    return None, None, None

        #face_on_image = face[0]

        #image, face, crop_face = futils.dlib.crop(
        #    image, face_on_image, self.up_ratio, self.down_ratio, self.width_ratio)
        np_image = np.array(image)
        # mask = self.face_parse.parse(ms.Tensor(cv2.resize(np_image, (512, 512))))
        mask = self.face_parse.parse(cv2.resize(np_image, (512, 512)))

        # obtain face parsing result
        # image = image.resize((512, 512), Image.ANTIALIAS)
        mask = ops.interpolate(
            mask.view(1, 1, 512, 512),
            (self.img_size, self.img_size),
            mode="nearest")
        
        # mask = ms.Tensor(mask,dtype=ms.uint8)
        mask = mask.astype(ms.uint8)
        # mask = mask.type(torch.uint8)
        mask = to_var(mask, requires_grad=False)  # .to(self.device)

        # detect landmark
        # landmarker = Landmarks()
        img = cv2.cvtColor(np.asarray(image),cv2.COLOR_RGB2BGR) 
        data = face_recognition.face_landmarks(img, model="large")
        values = [value for d in data for value in d.values()]
        landmark_array = [[num for pair in values for tup in pair for num in tup]]
        # landmark_array, pose_array, bbox_array = landmarker.inference(img, mode = 280)
        # landmarker.destory()
        if (len(landmark_array)==0):
            return None,None,None

        #print(len(landmark_array))
        lms = np.reshape(np.array(landmark_array[0]),(-1,2))
        lms[:,[1,0]] = lms[:,[0,1]]
        # print(lms[52:58])
        #lms = futils.dlib.landmarks(image, face) * self.img_size / image.width
        lms = lms*self.img_size/image.width
        lms = lms.round()

        mask, diff, rect, lms = self.process(mask, lms, device=self.device)
        image = image.resize((self.img_size, self.img_size), Image.ANTIALIAS)
        image = transform(image)
        image = ms.Tensor(image)
        real = to_var(image.unsqueeze(0))
        return [real, mask, diff, rect, lms], image, image
