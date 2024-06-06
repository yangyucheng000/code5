import mindspore as ms
import mindspore.nn as nn
from mindspore import ops

from ops.histogram_matching import histogram_matching

import os



class HistogramLoss(nn.Cell):
    def __init__(self):
        super(HistogramLoss, self).__init__()

    def de_norm(self, x):
        out = (x + 1) / 2
        return out.clamp(0, 1)

    def to_var(self, x, requires_grad=True):
        # if torch.cuda.is_available():
        #     x = x.cuda()
        if not requires_grad:
            #TODO Variable
            # return Variable(x, requires_grad=requires_grad)
            return ms.Tensor(x)
        else:
            # return Variable(x)
            return ms.Tensor(x)

    def construct(self, input_data, target_data, mask_src, mask_tar,mark=False):
        index_tmp = mask_src.unsqueeze(0).nonzero()
        x_A_index = index_tmp[:, 3]
        y_A_index = index_tmp[:, 4]
        index_tmp = mask_tar.unsqueeze(0).nonzero()
        x_B_index = index_tmp[:, 3]
        y_B_index = index_tmp[:, 4]

        input_data = (self.de_norm(input_data) * 255).squeeze()
        target_data = (self.de_norm(target_data) * 255).squeeze()
        mask_src = mask_src.broadcast_to(1, 3, mask_src.size(2), mask_src.size(2)).squeeze()
        mask_tar = mask_tar.broadcast_to(1, 3, mask_tar.size(2), mask_tar.size(2)).squeeze()
        input_masked = input_data * mask_src
        target_masked = target_data * mask_tar
        input_match = histogram_matching(
            input_masked, target_masked,
            [x_A_index, y_A_index, x_B_index, y_B_index],mark)
        input_match = self.to_var(input_match, requires_grad=False)
        #img_train_list=torch.cat([input_masked,input_match,target_masked],dim=2)
        #save_path = os.path.join('/data00/sunjingna/makeup/test', name+'.png')
        #save_image(img_train_list.data, save_path, normalize=True)
        loss = ops.l1_loss(input_masked, input_match)
        return loss
