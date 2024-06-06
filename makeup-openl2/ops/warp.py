# import torch
# import numpy as np
# import cv2
# import torch.nn.functional as F
# import torch.nn as nn
# from torchvision.utils import save_image
# import os.path as osp
# import os
# from torch.autograd import Variable


# def to_var( x, requires_grad=True):
#     if torch.cuda.is_available():
#         x = x.cuda()
#     if not requires_grad:
#         return Variable(x, requires_grad=requires_grad)
#     else:
#         return Variable(x)


# def de_norm( x):
#     out = (x + 1) / 2
#     return out.clamp(0, 1)


# def vis_train(img_train_list):
#     # saving training results
#     # mode = "train_vis"
#     img_train_list = torch.cat(img_train_list, dim=3)
#     result_path_train = osp.join("/opt/tiger/sjn_makeup/", "align")
#     if not osp.exists(result_path_train):
#         os.makedirs(result_path_train)
#     save_path = os.path.join(result_path_train, '1.jpg')
#     save_image(de_norm(img_train_list.data), save_path, normalize=True)



# def crop_face_best_fit(img_s, landmarks_s, img_r, landmarks_r, laplace_r):
#     landmarks_s = np.ravel(landmarks_s)
#     landmarks_r = np.ravel(landmarks_r)
#     if len(landmarks_s) < 1:
#         return None, None
#     affine_matrix = np.zeros((2, 3))
#     affine_matrix_inverse = np.zeros((2, 3))
#     src_vec, dst_vec = [], []
#     mean_sx, mean_sy, mean_dx, mean_dy = 0.0, 0.0, 0.0, 0.0
#     for i in range(int(len(landmarks_r) / 2)):
#         mean_sx += landmarks_r[2 * i]
#         mean_sy += landmarks_r[2 * i + 1]
#         mean_dx += landmarks_s[2 * i]
#         mean_dy += landmarks_s[2 * i + 1]
#     mean_sx /= (len(landmarks_r) / 2)
#     mean_sy /= (len(landmarks_r) / 2)
#     mean_dx /= (len(landmarks_r) / 2)
#     mean_dy /= (len(landmarks_r) / 2)

#     for i in range(int(len(landmarks_s) / 2)):
#         src_vec.append(landmarks_r[2 * i] - mean_sx)
#         src_vec.append(landmarks_r[2 * i + 1] - mean_sy)
#         dst_vec.append(landmarks_s[2 * i] - mean_dx)
#         dst_vec.append(landmarks_s[2 * i + 1] - mean_dy)
#     a, b, norm = 0.0, 0.0, 0.0
#     for i in range(int(len(landmarks_r) / 2)):
#         a += src_vec[2 * i] * dst_vec[2 * i] + src_vec[2 * i + 1] * dst_vec[2 * i + 1]
#         b += src_vec[2 * i] * dst_vec[2 * i + 1] - src_vec[2 * i + 1] * dst_vec[2 * i]
#         norm += src_vec[2 * i] * src_vec[2 * i] + src_vec[2 * i + 1] * src_vec[2 * i + 1]

#     a /= norm
#     b /= norm
#     mean_sx_trans = a * mean_sx - b * mean_sy
#     mean_sy_trans = b * mean_sx + a * mean_sy
#     affine_matrix[0, 0] = 1.0 / a
#     affine_matrix[0, 1] = -b
#     affine_matrix[1, 0] = b
#     affine_matrix[1, 1] = 1.0 / a
#     affine_matrix[0, 2] = (mean_dx - mean_sx_trans) / img_s.size(2)
#     affine_matrix[1, 2] = (mean_dy - mean_sy_trans) / img_s.size(3)
#     affine_matrix = torch.tensor(affine_matrix)
#     # face = cv2.warpAffine(img, affine_matrix, (crop_size, crop_size))
#     grid = F.affine_grid(affine_matrix.unsqueeze(0), img_r.size())
#     grid = grid.to("cuda").float()
#     output = F.grid_sample(img_r, grid)
#     output_laplace = F.grid_sample(laplace_r, grid)

#     return output, output_laplace


# def laplace_target_wrap(data_c, data_s, mask_c, mask_s, rect_c, rect_s, lms_c, lms_s):
#     # skin_s = torch.mean(torch.mean(data_s, dim=2,keepdim = True),dim=3,keepdim=True)
#     rect_c = rect_c.squeeze(0)
#     rect_s = rect_s.squeeze(0)
#     lms_c = lms_c.squeeze(0).cpu().numpy()
#     lms_s = lms_s.squeeze(0).cpu().numpy()
#     rect_c = rect_c / mask_c.size(2) * data_s.size(2)
#     rect_s = rect_s / mask_s.size(2) * data_s.size(2)
#     mask_c = F.interpolate(mask_c, size=(data_s.size(2), data_s.size(3)))
#     mask_s = F.interpolate(mask_s, size=(data_s.size(2), data_s.size(3)))
#     # erode_kernel = np.ones((10, 10), np.uint8)
#     # mask_c = torch.tensor(cv2.erode(mask_c.cpu().numpy().squeeze(), erode_kernel, iterations=1),
#     #                       dtype=torch.float32).unsqueeze(0).unsqueeze(0).to("cuda")
#     # mask_s = torch.tensor(cv2.erode(mask_s.cpu().numpy().squeeze(), erode_kernel, iterations=1),
#     #                       dtype=torch.float32).unsqueeze(0).unsqueeze(0).to("cuda")

#     kernel = torch.tensor(
#         np.broadcast_to(np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]]), (data_s.size(1), data_s.size(1), 3, 3)),
#         dtype=torch.float32).to("cuda")
#     laplace_c = F.conv2d(data_c, kernel, bias=None, stride=1, padding=1, dilation=1, groups=1)
#     laplace_s = F.conv2d(data_s, kernel, bias=None, stride=1, padding=1, dilation=1, groups=1)
#     laplace_c_ori = data_c * mask_c
#     laplace_s_ori = data_s * mask_s
#     laplace_s = laplace_s * mask_s
#     laplace_c = laplace_c * mask_c

#     s_w_left = int(rect_s[3]) - int(rect_s[1])
#     s_h_left = int(rect_s[2]) - int(rect_s[0])
#     s_w_right = int(rect_s[7]) - int(rect_s[5])
#     s_h_right = int(rect_s[6]) - int(rect_s[4])

#     c_w_left = int(rect_c[3]) - int(rect_c[1])
#     c_h_left = int(rect_c[2]) - int(rect_c[0])
#     c_w_right = int(rect_c[7]) - int(rect_c[5])
#     c_h_right = int(rect_c[6]) - int(rect_c[4])

#     center_c_left_x, center_c_left_y = int((rect_c[1] + rect_c[3]) // 2), int((rect_c[0] + rect_c[2]) // 2)
#     center_c_right_x, center_c_right_y = int((rect_c[5] + rect_c[7]) // 2), int((rect_c[4] + rect_c[6]) // 2)
#     center_s_left_x, center_s_left_y = int((rect_s[1] + rect_s[3]) // 2), int((rect_s[0] + rect_s[2]) // 2)
#     center_s_right_x, center_s_right_y = int((rect_s[5] + rect_s[7]) // 2), int((rect_s[4] + rect_s[6]) // 2)

#     rect_ori_c_left = laplace_c_ori[:, :, int(rect_c[1]):int(rect_c[3]), int(rect_c[0]):int(rect_c[2])]
#     rect_ori_c_right = laplace_c_ori[:, :, int(rect_c[5]):int(rect_c[7]), int(rect_c[4]):int(rect_c[6])]

#     rect_mask_c_left = mask_c[:, :, int(rect_c[1]):int(rect_c[3]), int(rect_c[0]):int(rect_c[2])]
#     rect_mask_c_right = mask_c[:, :, int(rect_c[5]):int(rect_c[7]), int(rect_c[4]):int(rect_c[6])]

#     rect_ori_s_left = laplace_s_ori[:, :, int(rect_s[1]):int(rect_s[3]), int(rect_s[0]):int(rect_s[2])]
#     rect_laplace_s_left = laplace_s[:, :, int(rect_s[1]):int(rect_s[3]), int(rect_s[0]):int(rect_s[2])]
#     rect_ori_s_right = laplace_s_ori[:, :, int(rect_s[5]):int(rect_s[7]), int(rect_s[4]):int(rect_s[6])]
#     rect_laplace_s_right = laplace_s[:, :, int(rect_s[5]):int(rect_s[7]), int(rect_s[4]):int(rect_s[6])]

#     c_ori = torch.zeros([data_c.size(0), data_c.size(1), 80, 80], dtype=torch.float).to("cuda")
#     c_mask = torch.zeros([data_c.size(0), data_c.size(1), 80, 80], dtype=torch.float).to("cuda")
#     s_ori = torch.zeros([data_s.size(0), data_s.size(1), 80, 80], dtype=torch.float).to("cuda")
#     s_laplace = torch.zeros([data_s.size(0), data_s.size(1), 80, 80], dtype=torch.float).to("cuda")
#     c_ori[:, :, 40 - c_w_left // 2:40 + c_w_left // 2 + c_w_left % 2,
#     40 - c_h_left // 2:40 + c_h_left // 2 + c_h_left % 2] = rect_ori_c_left
#     c_mask[:, :, 40 - c_w_left // 2:40 + c_w_left // 2 + c_w_left % 2,
#     40 - c_h_left // 2:40 + c_h_left // 2 + c_h_left % 2] = rect_mask_c_left

#     s_ori[:, :, 40 - s_w_left // 2:40 + s_w_left // 2 + s_w_left % 2,
#     40 - s_h_left // 2:40 + s_h_left // 2 + s_h_left % 2] = rect_ori_s_left
#     s_laplace[:, :, 40 - s_w_left // 2:40 + s_w_left // 2 + s_w_left % 2,
#     40 - s_h_left // 2:40 + s_h_left // 2 + s_h_left % 2] = rect_laplace_s_left

#     # skin_s = torch.mean(torch.mean(s_ori, dim=2,keepdim = True),dim=3,keepdim=True)
#     # index_ori = torch.nonzero(s_ori).cpu().numpy()
#     # s_ori_clone=torch.ones([data_s.size(0),data_s.size(1),80,80],dtype=torch.float).to("cuda") *skin_s
#     # s_ori_clone[:,0,index_ori[:,2],index_ori[:,3]]=s_ori[:,0,index_ori[:,2],index_ori[:,3]]
#     # s_ori_clone[:,1,index_ori[:,2],index_ori[:,3]]=s_ori[:,1,index_ori[:,2],index_ori[:,3]]
#     # s_ori_clone[:,2,index_ori[:,2],index_ori[:,3]]=s_ori[:,2,index_ori[:,2],index_ori[:,3]]

#     lms_c_left = lms_c[42:48] - np.array([center_c_left_x, center_c_left_y])
#     lms_c_right = lms_c[36:42] - np.array([center_c_right_x, center_c_right_y])
#     lms_s_left = lms_s[42:48] - np.array([center_s_left_x, center_s_left_y])
#     lms_s_right = lms_s[36:42] - np.array([center_s_right_x, center_s_right_y])

#     warp_s_ori_left, warp_s_laplace_left = crop_face_best_fit(c_ori, lms_c_left, s_ori, lms_s_left, s_laplace)

#     # warp_s_ori_left = warp_s_ori_left - warp_s_laplace_left

#     warp_s_ori_left = warp_s_ori_left * c_mask
#     warp_s_laplace_left = warp_s_laplace_left * c_mask
#     # vis_train([c_ori,warp_s_ori_left, s_ori])
#     #    index_ori = torch.nonzero(warp_s_ori_left).cpu().numpy()
#     #    index_x = index_ori[:,2]+center_c_left_x-40
#     #    index_y = index_ori[:,3]+center_c_left_y-40
#     #    data_c_1=torch.zeros([data_c.size(0),data_c.size(1),data_c.size(2),data_c.size(3)],dtype=torch.float).to("cuda") + data_c
#     #    data_c_1[:,0,index_x,index_y]=s_ori[:,0,index_ori[:,2],index_ori[:,3]]
#     #    data_c_1[:,1,index_x,index_y]=s_ori[:,1,index_ori[:,2],index_ori[:,3]]
#     #    data_c_1[:,2,index_x,index_y]=s_ori[:,2,index_ori[:,2],index_ori[:,3]]

#     c_ori = torch.zeros([data_c.size(0), data_c.size(1), 80, 80], dtype=torch.float).to("cuda")
#     c_mask = torch.zeros([data_c.size(0), data_c.size(1), 80, 80], dtype=torch.float).to("cuda")
#     s_ori = torch.zeros([data_s.size(0), data_s.size(1), 80, 80], dtype=torch.float).to("cuda")
#     s_laplace = torch.zeros([data_s.size(0), data_s.size(1), 80, 80], dtype=torch.float).to("cuda")
#     c_ori[:, :, 40 - c_w_right // 2:40 + c_w_right // 2 + c_w_right % 2,
#     40 - c_h_right // 2:40 + c_h_right // 2 + c_h_right % 2] = rect_ori_c_right
#     c_mask[:, :, 40 - c_w_right // 2:40 + c_w_right // 2 + c_w_right % 2,
#     40 - c_h_right // 2:40 + c_h_right // 2 + c_h_right % 2] = rect_mask_c_right
#     s_ori[:, :, 40 - s_w_right // 2:40 + s_w_right // 2 + s_w_right % 2,
#     40 - s_h_right // 2:40 + s_h_right // 2 + s_h_right % 2] = rect_ori_s_right
#     s_laplace[:, :, 40 - s_w_right // 2:40 + s_w_right // 2 + s_w_right % 2,
#     40 - s_h_right // 2:40 + s_h_right // 2 + s_h_right % 2] = rect_laplace_s_right

#     # skin_s = torch.mean(torch.mean(s_ori, dim=2,keepdim = True),dim=3,keepdim=True)
#     # index_ori = torch.nonzero(s_ori).cpu().numpy()
#     # s_ori_clone=torch.ones([data_s.size(0),data_s.size(1),80,80],dtype=torch.float).to("cuda") * skin_s
#     # s_ori_clone[:,0,index_ori[:,2],index_ori[:,3]]=s_ori[:,0,index_ori[:,2],index_ori[:,3]]
#     # s_ori_clone[:,1,index_ori[:,2],index_ori[:,3]]=s_ori[:,1,index_ori[:,2],index_ori[:,3]]
#     # s_ori_clone[:,2,index_ori[:,2],index_ori[:,3]]=s_ori[:,2,index_ori[:,2],index_ori[:,3]]

#     warp_s_ori_right, warp_s_laplace_right = crop_face_best_fit(c_ori, lms_c_right, s_ori, lms_s_right, s_laplace)

#     # warp_s_ori_right = warp_s_ori_right - warp_s_laplace_right

#     warp_s_ori_right = warp_s_ori_right * c_mask
#     warp_s_laplace_right = warp_s_laplace_right * c_mask
#     # print(warp_s_laplace_right.shape, warp_s_ori_right.shape, s_ori.shape)
#     # vis_train([c_ori,warp_s_ori_right, s_ori])
#     #    index_ori = torch.nonzero(warp_s_ori_right).cpu().numpy()
#     #    index_x = index_ori[:,2]+center_c_right_x-40
#     #    index_y = index_ori[:,3]+center_c_right_y-40
#     #    data_c_1[:,0,index_x,index_y]=s_ori[:,0,index_ori[:,2],index_ori[:,3]]
#     #    data_c_1[:,1,index_x,index_y]=s_ori[:,1,index_ori[:,2],index_ori[:,3]]
#     #    data_c_1[:,2,index_x,index_y]=s_ori[:,2,index_ori[:,2],index_ori[:,3]]
#     #    #vis_train([data_c_1,data_c, data_s])

#     align_s = torch.zeros_like(laplace_c, dtype=torch.float).to("cuda")
#     align_s_ori = torch.zeros_like(laplace_c_ori, dtype=torch.float).to("cuda")

#     center_c_left_x, center_c_left_y = int((rect_c[1] + rect_c[3]) // 2), int((rect_c[0] + rect_c[2]) // 2)
#     center_c_right_x, center_c_right_y = int((rect_c[5] + rect_c[7]) // 2), int((rect_c[4] + rect_c[6]) // 2)
#     # k=align_s[:,:,center_c_left_x-w_left//2:center_c_left_x+w_left//2+w_left%2, center_c_left_y-h_left//2:center_c_left_y+h_left//2+h_left%2]
#     # print(k.shape)
#     # print(center_c_right_x)
#     # vis_train([data_c])

#     align_s[:, :, center_c_left_x - 40:center_c_left_x + 40,
#     center_c_left_y - 40:center_c_left_y + 40] = warp_s_laplace_left
#     align_s[:, :, max(0, center_c_right_x - 40):center_c_right_x + 40,
#     center_c_right_y - 40:center_c_right_y + 40] = warp_s_laplace_right[:, :,
#                                                    abs(center_c_right_x - 40 - max(0, center_c_right_x - 40)):, :]

#     align_s_ori[:, :, center_c_left_x - 40:center_c_left_x + 40,
#     center_c_left_y - 40:center_c_left_y + 40] = warp_s_ori_left
#     align_s_ori[:, :, max(0, center_c_right_x - 40):center_c_right_x + 40,
#     center_c_right_y - 40:center_c_right_y + 40] = warp_s_ori_right[:, :,
#                                                    abs(center_c_right_x - 40 - max(0, center_c_right_x - 40)):, :]


#     return  align_s, align_s_ori, laplace_c, laplace_c_ori



# class WarppixlLoss(nn.Module):
#     def __init__(self):
#         super(WarppixlLoss, self).__init__()

#     def forward(self, data_c, mask, warp_ori, warp_laplace,flag = True):
#         # mask_c_skin, mask_s_skin = mask_c[:, 3, :, :, :], mask_s[:, 3, :, :, :]
#         # mask_c, mask_s = mask_c[:, 0:3, :, :, :], mask_s[:, 0:3, :, :, :]
#         # align_laplace, align_ori, data_c_laplace, data_c_ori = laplace_target_wrap(data_c, data_s, mask_c_skin, mask_s_skin, rect_c, rect_s, lms_c, lms_s)
#         # data_c_laplace = data_c_laplace * mask_c[:,2]
#         # data_c_ori = data_c_ori * mask_c[:,2]
#         # align_ori = to_var(align_ori,requires_grad=False)
#         # align_laplace = to_var(align_laplace, requires_grad=False)
#         # # vis_train([data_c, data_s, data_c_ori ,align_ori, data_c_laplace,align_laplace])

#         # loss = F.l1_loss(data_c_ori , align_ori) + F.l1_loss(data_c_laplace , align_laplace) *10
#         # return loss

#         kernel = torch.tensor(
#             np.broadcast_to(np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]]), (data_c.size(1), data_c.size(1), 3, 3)),
#             dtype=torch.float32).to("cuda")
#         laplace_c = F.conv2d(data_c, kernel, bias=None, stride=1, padding=1, dilation=1, groups=1)
#         data_c_ori = data_c * mask
#         data_c_laplace = laplace_c * mask
#         index = torch.nonzero(warp_ori).cpu().numpy()
#         index_x = index[:,2]
#         index_y = index[:,3]
#         mask_final = torch.zeros([warp_ori.size(0), warp_ori.size(1), warp_ori.size(2), warp_ori.size(3)],dtype=torch.float).to("cuda")
#         mask_final[:,0,index_x,index_y]=1
#         mask_final[:,1,index_x,index_y]=1
#         mask_final[:,2,index_x,index_y]=1

        
#         # vis_train([data_c_ori,warp_ori, data_c_ori+warp_ori,data_c_laplace, warp_laplace])
#         if flag:
#             loss = F.l1_loss(data_c_ori*mask_final , warp_ori*mask_final)*2+ F.l1_loss(data_c_laplace*mask_final , warp_laplace*mask_final) *10
#         else:
#             loss = F.l1_loss(data_c_ori , warp_ori)*2 + F.l1_loss(data_c_laplace , warp_laplace) *10
#         return loss
