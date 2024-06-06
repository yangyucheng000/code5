#!/usr/bin/python
# -*- encoding: utf-8 -*-
import os
import os.path as osp
from typing import Type
pwd = osp.split(osp.realpath(__file__))[0]

import time
import datetime
from PIL import Image

import mindspore as ms
from mindspore import nn
from mindspore.dataset.vision import ToPIL
from mindspore.common.initializer import initializer,XavierNormal
from mindcv.models import vgg16
# TODO save_image (train)
# TODO init
# TODO Variable

from ops.loss_added import GANLoss
from ops.histogram_loss import HistogramLoss
from ops.smmoth_loss import SmmothLoss
import tools.plot as plot_fig
from .preprocess import PreProcess
from concern.track import Track
import numpy as np


class Solver(Track):
    def __init__(self, config, device="cpu", data_loader=None, inference=False):
        self.config = config
        print('no local d && no warp loss')
        if self.config.MODEL.NET == 'net_dran':
            from . import net_dran as net
        else:
            raise TypeError(self.config.net)
        self.G = net.Generator(config)
        # print(self.G)
        # for name,param in self.G.parameters_and_names():
        #     print(name)
        if inference:
            ms.load_param_into_net(self.G, ms.load_checkpoint(inference))
            self.G.set_train(False)
            return

        self.start_time = time.time()
        self.checkpoint = config.MODEL.WEIGHTS
        self.log_path = config.LOG.LOG_PATH
        self.result_path = os.path.join(self.log_path, config.LOG.VIS_PATH)
        self.snapshot_path = os.path.join(self.log_path, config.LOG.SNAPSHOT_PATH)
        self.log_step = config.LOG.LOG_STEP
        self.vis_step = config.LOG.VIS_STEP
        #TODO torch.cuda.device_count()
        self.snapshot_step = config.LOG.SNAPSHOT_STEP // ms.communication.get_group_size()  # torch.cuda.device_count()

        # Data loader
        self.data_loader_train = data_loader
        self.img_size = config.DATA.IMG_SIZE

        self.num_epochs = config.TRAINING.NUM_EPOCHS
        self.num_epochs_decay = config.TRAINING.NUM_EPOCHS_DECAY
        self.g_lr = config.TRAINING.G_LR
        self.d_lr = config.TRAINING.D_LR
        self.g_step = config.TRAINING.G_STEP
        self.beta1 = config.TRAINING.BETA1
        self.beta2 = config.TRAINING.BETA2

        self.lambda_idt      = config.LOSS.LAMBDA_IDT
        self.lambda_A        = config.LOSS.LAMBDA_A
        self.lambda_B        = config.LOSS.LAMBDA_B
        self.lambda_his_lip  = config.LOSS.LAMBDA_HIS_LIP
        self.lambda_his_skin = config.LOSS.LAMBDA_HIS_SKIN
        self.lambda_his_eye  = config.LOSS.LAMBDA_HIS_EYE
        self.lambda_vgg      = config.LOSS.LAMBDA_VGG


        # Hyper-parameteres
        self.d_conv_dim = config.MODEL.D_CONV_DIM
        self.d_repeat_num = config.MODEL.D_REPEAT_NUM
        self.norm = config.MODEL.NORM

        self.device = device

        self.build_model()
        super(Solver, self).__init__()

    # For generator
    def weights_init_xavier(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            #TODO 是否一样
            # init.xavier_normal(m.weight.data, gain=1.0)
            for name, param in m.parameters_and_names():
                if 'weight' in name:
                    param.set_data(initializer(XavierNormal(), param.shape, param.dtype))
        elif classname.find('Linear') != -1:
            for name, param in m.parameters_and_names():
                if 'weight' in name:
                    param.set_data(initializer(XavierNormal(), param.shape, param.dtype))

    def print_network(self, model, name):
        num_params = 0
        for p in model.get_parameters():
            num_params += p.numel()
        # print(name)
        # print(model)
        # print("The number of parameters: {}".format(num_params))

    def de_norm(self, x):
        out = (x + 1) / 2
        return out.clamp(0, 1)

    def build_model(self):
        # self.G = net.Generator()
        if self.config.MODEL.NET == 'net_dran':
            from . import net_dran as net
        else:
            raise TypeError(self.config.net)
        self.D_A = net.Discriminator(self.img_size, self.d_conv_dim, self.d_repeat_num, self.norm)
        self.D_B = net.Discriminator(self.img_size, self.d_conv_dim, self.d_repeat_num, self.norm)


        self.G.apply(self.weights_init_xavier)
        self.D_A.apply(self.weights_init_xavier)
        self.D_B.apply(self.weights_init_xavier)

        self.load_checkpoint()
        self.criterionL1 = nn.L1Loss()
        self.criterionL2 = nn.MSELoss()
        self.criterionGAN = GANLoss(use_lsgan=True, tensor=ms.Tensor)
        self.criterionBCE = nn.BCELoss()
        
        #TODO right？
        # self.vgg = net.vgg16(pretrained=True)
        self.vgg = vgg16(pretrained=True)
        self.criterionHis = HistogramLoss()
        self.criterionLaplace = SmmothLoss()

        # Optimizers
        self.g_optimizer = nn.Adam(self.G.get_parameters(), self.g_lr, [self.beta1, self.beta2])
        self.d_A_optimizer = nn.Adam(filter(lambda p: p.requires_grad, self.D_A.get_parameters()), self.d_lr, [self.beta1, self.beta2])
        self.d_B_optimizer = nn.Adam(filter(lambda p: p.requires_grad, self.D_B.get_parameters()), self.d_lr, [self.beta1, self.beta2])

        # Print networks
        self.print_network(self.G, 'G')
        self.print_network(self.D_A, 'D_A')
        self.print_network(self.D_B, 'D_B')

        #TODO DataParallel
        # if torch.cuda.is_available():
        #     self.device = "cuda"
        #     if torch.cuda.device_count() > 1:
        #         self.G = nn.DataParallel(self.G)
        #         self.D_A = nn.DataParallel(self.D_A)
        #         self.D_B = nn.DataParallel(self.D_B)
        #         self.vgg = nn.DataParallel(self.vgg)
        #         self.criterionHis = nn.DataParallel(self.criterionHis)
        #         self.criterionGAN = nn.DataParallel(self.criterionGAN)
        #         self.criterionL1 = nn.DataParallel(self.criterionL1)
        #         self.criterionL2 = nn.DataParallel(self.criterionL2)
        #         self.criterionLaplace = nn.DataParallel(self.criterionLaplace)

        #     self.G.cuda()
        #     self.vgg.cuda()
        #     self.criterionHis.cuda()
        #     self.criterionGAN.cuda()
        #     self.criterionL1.cuda()
        #     self.criterionL2.cuda()
        #     self.criterionLaplace.cuda()
        #     self.D_A.cuda()
        #     self.D_B.cuda()

    # def load_checkpoint(self):
    #     G_path = self.checkpoint + '_G.pth'
    #     if os.path.exists(G_path):
    #         self.G.load_state_dict(torch.load(G_path))
    #         print('loaded trained generator {}..!'.format(G_path))
    #     D_A_path = self.checkpoint + '_D_A.pth'
    #     if os.path.exists(D_A_path):
    #         self.D_A.load_state_dict(torch.load(D_A_path))
    #         print('loaded trained discriminator A {}..!'.format(D_A_path))

    #     D_B_path = self.checkpoint + '_D_B.pth'
    #     if os.path.exists(D_B_path):
    #         self.D_B.load_state_dict(torch.load(D_B_path))
    #         print('loaded trained discriminator B {}..!'.format(D_B_path))

    def generate(self, org_A, ref_B, lms_A=None, lms_B=None, mask_A=None, mask_B=None, 
                 diff_A=None, diff_B=None, rect_A=None, rect_B=None, lms_A1=None, lms_B1=None, gamma=None, beta=None, ret=False):
        """org_A is content, ref_B is style"""
        res = self.G(org_A, ref_B, mask_A, mask_B, diff_A, diff_B, rect_A, rect_B, lms_A1, lms_B1, gamma, beta, ret)
        return res

    # mask attribute: 0:background 1:face 2:left-eyebrown 3:right-eyebrown 4:left-eye 5: right-eye 6: nose
    # 7: upper-lip 8: teeth 9: under-lip 10:hair 11: left-ear 12: right-ear 13: neck

    def test(self, real_A, mask_A, diff_A, rect_A, lms_A, real_B, mask_B, diff_B, rect_B, lms_B):
        cur_prama = None
        # with torch.no_grad():
        #     fake_A, mid_results = self.generate(real_A.to("cuda"), real_B.to("cuda"), None, None, mask_A.to("cuda"), mask_B.to("cuda"), 
        #                            diff_A.to("cuda"), diff_B.to("cuda"), rect_A.to("cuda"), rect_B.to("cuda"), lms_A.to("cuda"), lms_B.to("cuda"))
        fake_A, mid_results = self.generate(real_A, real_B, None, None, mask_A, mask_B, 
                                    diff_A, diff_B, rect_A, rect_B, lms_A, lms_B)
        fake_A = fake_A.squeeze(0)

        # normalize
        min_, max_ = fake_A.min(), fake_A.max()
        # fake_A.add_(-min_).div_(max_ - min_ + 1e-5)
        fake_A = (fake_A.add(-min_)).div(max_ - min_ + 1e-5)
        img=[]
        A = ((fake_A.asnumpy().transpose(1,2,0))*255).astype(np.uint8)
        # img.append(ToPILImage()(fake_A.cpu()))
        img.append(Image.fromarray(A))
        return Image.fromarray(A), img, mid_results


    # def train(self):
    #     # The number of iterations per epoch
    #     self.iters_per_epoch = len(self.data_loader_train)
    #     # Start with trained model if exists
    #     g_lr = self.g_lr
    #     d_lr = self.d_lr
    #     start = 0

    #     for self.e in range(start, self.num_epochs):
    #         for self.i, (source_input, reference_input) in enumerate(self.data_loader_train):
    #             # image, mask, dist
    #             image_s, image_r = source_input[0].to(self.device), reference_input[0].to(self.device)
    #             mask_s, mask_r = source_input[1].to(self.device), reference_input[1].to(self.device) 
    #             dist_s, dist_r = source_input[2].to(self.device), reference_input[2].to(self.device)
    #             rect_s, rect_r = source_input[3].to(self.device), reference_input[3].to(self.device)
    #             lms_s, lms_r = source_input[4].to(self.device), reference_input[4].to(self.device)
    #             self.track("data")
    #             # import pdb;pdb.set_trace()
    #             # ================== Train D ================== #
    #             # training D_A, D_A aims to distinguish class B
    #             # Real
    #             out = self.D_A(image_r)
    #             #print("out",out.shape)
    #             self.track("D_A")
    #             d_loss_real = self.criterionGAN(out, True)
    #             #print("loss",d_loss_real)
    #             self.track("D_A_loss")
    #             # Fake
    #             fake_A,_= self.G(image_s, image_r, mask_s, mask_r, dist_s, dist_r, rect_s, rect_r, lms_s, lms_r,epoch=self.e, warp=False)
    #             self.track("G")
    #             fake_A = Variable(fake_A.data).detach()
    #             out = self.D_A(fake_A)
    #             self.track("D_A_2")
    #             d_loss_fake =  self.criterionGAN(out, False)
    #             self.track("D_A_loss_2")

    #             # Backward + Optimize
    #             d_loss = (d_loss_real.mean() + d_loss_fake.mean()) * 0.5 *2
    #             self.d_A_optimizer.zero_grad()
    #             d_loss.backward(retain_graph=False)
    #             self.d_A_optimizer.step()

    #             # Logging
    #             self.loss = {}
    #             self.loss['D-A-loss_real'] = d_loss_real.mean().item()

    #             # training D_B, D_B aims to distinguish class A
    #             # Real
    #             out = self.D_B(image_s)
    #             d_loss_real = self.criterionGAN(out, True)
    #             # Fake
    #             self.track("G-before")
    #             fake_B,_ = self.G(image_r, image_s, mask_r, mask_s, dist_r, dist_s, rect_r, rect_s, lms_r, lms_s,epoch=self.e, warp=False)
    #             self.track("G-2")
    #             fake_B = Variable(fake_B.data).detach()
    #             out = self.D_B(fake_B)
    #             d_loss_fake =  self.criterionGAN(out, False)

    #             # Backward + Optimize
    #             d_loss = (d_loss_real.mean() + d_loss_fake.mean()) * 0.5 *2
    #             self.d_B_optimizer.zero_grad()
    #             d_loss.backward(retain_graph=False)
    #             self.d_B_optimizer.step()

    #             # Logging
    #             self.loss['D-B-loss_real'] = d_loss_real.mean().item()


                
    #             # ================== Train G ================== #
    #             if (self.i + 1) % self.g_step == 0:
    #                 # identity loss
    #                 assert self.lambda_idt > 0
                    
    #                 # G should be identity if ref_B or org_A is fed
    #                 idt_A,_ = self.G(image_s, image_s, mask_s, mask_s, dist_s, dist_s, rect_s, rect_s, lms_s, lms_s,epoch=self.e, warp=False)
    #                 idt_B,_ = self.G(image_r, image_r, mask_r, mask_r, dist_r, dist_r, rect_r, rect_r, lms_r, lms_r,epoch=self.e, warp=False)
    #                 loss_idt_A = self.criterionL1(idt_A, image_s) * self.lambda_A * self.lambda_idt
    #                 loss_idt_B = self.criterionL1(idt_B, image_r) * self.lambda_B * self.lambda_idt
    #                 # loss_idt
    #                 loss_idt = (loss_idt_A + loss_idt_B) * 0.5
    #                 # loss_idt = loss_idt_A * 0.5
    #                 # self.track("Identical")

    #                 # GAN loss D_A(G_A(A))
    #                 # fake_A in class B, 
    #                 fake_A,warp_A = self.G(image_s, image_r, mask_s, mask_r, dist_s, dist_r, rect_s, rect_r, lms_s, lms_r,epoch=self.e, warp=False)
    #                 pred_fake = self.D_A(fake_A)
    #                 g_A_loss_adv = self.criterionGAN(pred_fake, True)

    #                 # GAN loss D_B(G_B(B))
    #                 fake_B,warp_B= self.G(image_r, image_s, mask_r, mask_s, dist_r, dist_s, rect_r, rect_s, lms_r, lms_s,epoch=self.e, warp=False)
    #                 pred_fake = self.D_B(fake_B)
    #                 g_B_loss_adv = self.criterionGAN(pred_fake, True)

    #                 # self.track("Generator forward")

    #                 # color_histogram loss
    #                 g_A_loss_his = 0
    #                 g_B_loss_his = 0
    #                 g_A_lip_loss_his = self.criterionHis(
    #                     fake_A, image_r, mask_s[:, 0], mask_r[:, 0]
    #                 ) * self.lambda_his_lip*3
    #                 g_B_lip_loss_his = self.criterionHis(
    #                     fake_B, image_s, mask_r[:, 0], mask_s[:, 0]
    #                 ) * self.lambda_his_lip*3
    #                 g_A_loss_his += g_A_lip_loss_his
    #                 g_B_loss_his += g_B_lip_loss_his

    #                 g_A_skin_loss_his = self.criterionHis(
    #                         fake_A, image_r, mask_s[:, 1]+mask_s[:,6]+mask_s[:,3], mask_r[:, 1]+mask_r[:,6]+mask_r[:,3]
    #                 ) * self.lambda_his_skin
    #                 g_B_skin_loss_his = self.criterionHis(
    #                         fake_B, image_s, mask_r[:, 1]+mask_r[:,6]+mask_r[:,3], mask_s[:, 1]+mask_s[:,6]+mask_s[:,3]
    #                 ) * self.lambda_his_skin
    #                 g_A_loss_his += g_A_skin_loss_his
    #                 g_B_loss_his += g_B_skin_loss_his

    #                 g_A_eye_loss_his = self.criterionHis(
    #                   fake_A, image_r, mask_s[:, 6], mask_r[:, 6]
    #                 ) * self.lambda_his_eye
    #                 g_B_eye_loss_his = self.criterionHis(
    #                   fake_B, image_s, mask_r[:, 6], mask_s[:, 6]
    #                 ) * self.lambda_his_eye
    #                 g_A_loss_his += g_A_eye_loss_his*1
    #                 g_B_loss_his += g_B_eye_loss_his*1
                    

    #                 g_A_nose_loss_his = self.criterionHis(
    #                    fake_A, image_r, mask_s[:, 4], mask_r[:, 4],True
    #                 ) * self.lambda_his_skin*3
    #                 g_B_nose_loss_his = self.criterionHis(
    #                    fake_B, image_s, mask_r[:, 4], mask_s[:, 4],True
    #                 ) * self.lambda_his_skin*3
    #                 g_A_loss_his += g_A_nose_loss_his
    #                 g_B_loss_his += g_B_nose_loss_his

    #                 # cycle loss
    #                 rec_A,_ = self.G(fake_A, image_s, mask_s, mask_s, dist_s, dist_s, rect_s, rect_s, lms_s, lms_s,epoch=self.e, warp=False)
    #                 rec_B,_= self.G(fake_B, image_r, mask_r, mask_r, dist_r, dist_r, rect_r, rect_r, lms_r, lms_r,epoch=self.e, warp=False)

    #                 g_loss_rec_A = self.criterionL1(rec_A, image_s) * self.lambda_A
    #                 g_loss_rec_B = self.criterionL1(rec_B, image_r) * self.lambda_B
    #                 # self.track("Generator recover")

    #                 image_s_gray = (image_s[:,0]*0.299+image_s[:,1]*0.587+image_s[0,2]*0.114).unsqueeze(0).repeat(1,3,1,1)
    #                 vgg_s = self.vgg(image_s_gray)
    #                 vgg_s = Variable(vgg_s.data).detach()
    #                 fake_A_gray = (fake_A[:,0]*0.299+fake_A[:,1]*0.587+fake_A[0,2]*0.114).unsqueeze(0).repeat(1,3,1,1)
    #                 vgg_fake_A = self.vgg(fake_A_gray)
    #                 g_loss_A_vgg = self.criterionL2(vgg_fake_A, vgg_s) * self.lambda_A * self.lambda_vgg*200
    #                 # self.track("Generator vgg")
                    
    #                 image_r_gray = (image_r[:,0]*0.299+image_r[:,1]*0.587+image_r[0,2]*0.114).unsqueeze(0).repeat(1,3,1,1)
    #                 vgg_r = self.vgg(image_r_gray)
    #                 vgg_r = Variable(vgg_r.data).detach()
    #                 fake_B_gray = (fake_B[:,0]*0.299+fake_B[:,1]*0.587+fake_B[0,2]*0.114).unsqueeze(0).repeat(1,3,1,1)
    #                 vgg_fake_B = self.vgg(fake_B_gray)
    #                 g_loss_B_vgg = self.criterionL2(vgg_fake_B, vgg_r) * self.lambda_B * self.lambda_vgg*200

    #                 loss_rec = (g_loss_rec_A + g_loss_rec_B + g_loss_A_vgg + g_loss_B_vgg) * 0.5

    #                 # Combined loss
    #                 g_loss = (g_A_loss_adv + g_B_loss_adv + loss_rec + loss_idt + g_A_loss_his + g_B_loss_his).mean()

    #                 self.g_optimizer.zero_grad()
    #                 g_loss.backward(retain_graph=False)
    #                 self.g_optimizer.step()
    #                 # self.track("Generator backward")

    #                 # Logging
    #                 self.loss['G-A-loss-adv'] = g_A_loss_adv.mean().item()
    #                 self.loss['G-B-loss-adv'] = g_B_loss_adv.mean().item()
    #                 self.loss['G-loss-org'] = g_loss_rec_A.mean().item()
    #                 self.loss['G-loss-ref'] = g_loss_rec_B.mean().item()
    #                 self.loss['G-loss-idt'] = loss_idt.mean().item()
    #                 self.loss['G-loss-img-rec'] = (g_loss_rec_A + g_loss_rec_B).mean().item()
    #                 self.loss['G-loss-vgg-rec'] = (g_loss_A_vgg + g_loss_B_vgg).mean().item()
    #                 self.loss['G-loss-img-rec'] = g_loss_rec_A.mean().item()
    #                 self.loss['G-loss-vgg-rec'] = g_loss_A_vgg.mean().item()

    #                 self.loss['G-A-loss-his'] = g_A_loss_his.mean().item()


    #             # Print out log info
    #             if (self.i + 1) % self.log_step == 0:
    #                 self.log_terminal()

    #             #save the images
    #             if (self.i) % self.vis_step == 0:
    #                 print("Saving middle output...")
    #                 print('model is ', self.config)
    #                 self.vis_train([image_s, image_r, fake_A, rec_A, mask_s[:, 0:3, 0], mask_s[:,3:6,0], mask_r[:, 0:3, 0], mask_r[:, 2:5,0]])

    #             # Save model checkpoints
    #             if (self.i) % self.snapshot_step == 0:
    #                 self.save_models()

    #         # Decay learning rate
    #         if (self.e+1) > (self.num_epochs - self.num_epochs_decay):
    #             g_lr -= (self.g_lr / float(self.num_epochs_decay))
    #             d_lr -= (self.d_lr / float(self.num_epochs_decay))
    #             self.update_lr(g_lr, d_lr)
    #             print('Decay learning rate to g_lr: {}, d_lr:{}.'.format(g_lr, d_lr))

    # def update_lr(self, g_lr, d_lr):
    #     for param_group in self.g_optimizer.param_groups:
    #         param_group['lr'] = g_lr
    #     for param_group in self.d_A_optimizer.param_groups:
    #         param_group['lr'] = d_lr
    #     for param_group in self.d_B_optimizer.param_groups:
    #         param_group['lr'] = d_lr

    # def save_models(self):
    #     if not osp.exists(self.snapshot_path):
    #         os.makedirs(self.snapshot_path)
    #     torch.save(
    #         self.G.state_dict(),
    #         os.path.join(
    #             self.snapshot_path, '{}_{}_G.pth'.format(self.e + 1, self.i + 1)))
    #     torch.save(
    #         self.D_A.state_dict(),
    #         os.path.join(
    #             self.snapshot_path, '{}_{}_D_A.pth'.format(self.e + 1, self.i + 1)))
    #     torch.save(
    #         self.D_B.state_dict(),
    #         os.path.join(
    #             self.snapshot_path, '{}_{}_D_B.pth'.format(self.e + 1, self.i + 1)))

    # def vis_train(self, img_train_list):
    #     # saving training results
    #     mode = "train_vis"
    #     img_train_list = torch.cat(img_train_list, dim=3)
    #     result_path_train = osp.join(self.result_path, mode)
    #     if not osp.exists(result_path_train):
    #         os.makedirs(result_path_train)
    #     save_path = os.path.join(result_path_train, '{}_{}_fake.jpg'.format(self.e, self.i))
    #     save_image(self.de_norm(img_train_list.data), save_path, normalize=True)

    # def log_terminal(self):
    #     elapsed = time.time() - self.start_time
    #     elapsed = str(datetime.timedelta(seconds=elapsed))

    #     log = "Elapsed [{}], Epoch [{}/{}], Iter [{}/{}]".format(
    #         elapsed, self.e+1, self.num_epochs, self.i+1, self.iters_per_epoch)

    #     for tag, value in self.loss.items():
    #         log += ", {}: {:.4f}".format(tag, value)
    #     print(log)

    # def to_var(self, x, requires_grad=True):
    #     if torch.cuda.is_available():
    #         x = x.cuda()
    #     if not requires_grad:
    #         return Variable(x, requires_grad=requires_grad)
    #     else:
    #         return Variable(x)
