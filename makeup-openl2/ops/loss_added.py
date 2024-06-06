import mindspore as ms
from mindspore import nn
from mindspore import ops

class GANLoss(nn.Cell):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=ms.Tensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                #TODO 这样改对吗？
                fill = ops.FillV2()
                real_tensor = fill(self.Tensor(input.shape,ms.float32), ms.Tensor(self.real_label,ms.float32))
                # real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                #TODO Variable
                self.real_label_var = real_tensor
                # self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fill = ops.FillV2()
                fake_tensor = fill(self.Tensor(input.shape,ms.float32), ms.Tensor(self.fake_label,ms.float32))
                # fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                #TODO
                self.fake_label_var = fake_tensor
                # self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def construct(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        #print("target",target_tensor.shape)
        return self.loss(input, target_tensor)
