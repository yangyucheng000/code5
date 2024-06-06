import os
os.environ['CUDA_VISIBLE_DEVICES'] = '4'
import warnings
import torch

class DefaultConfig(object):
    load_img_path = None  # load image model path
    load_txt_path = None  # load txt model path

    # data parameters
    data_path = ''
    pretrain_model_path = './data/imagenet-vgg-f.mat'
    training_size = 8072
    query_size = 2000
    database_size = 18015
    batch_size = 128
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # hyper-parameters
    max_epoch = 500
    gamma = 1
    eta = 1
    alpha = 0.05
    beta = 0.05
    bit = 64  # final binary code length
    lr = 10 ** (-1.5)  # initial learning rate
    endecoder_lr = 10 ** (-2)
    use_gpu = True

    valid = True

    print_freq = 2  # print info every N epoch

    result_dir = 'result'

    def parse(self, kwargs):
        """
        update configuration by kwargs.
        """
        for k, v in kwargs.items():
            if not hasattr(self, k):
                warnings.warn("Waning: opt has no attribute %s" % k)
            setattr(self, k, v)

        print('User config:')
        for k, v in self.__dict__.items():
            if not k.startswith('__'):
                print(k, getattr(self, k))


opt = DefaultConfig()
