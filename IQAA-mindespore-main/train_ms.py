import ast
import argparse
import os
import mindspore.nn as nn
from mindspore.communication.management import init, get_rank
from mindspore import dataset as de
from mindspore import context
from mindspore import Tensor
# from mindspore.train import Model
from mindspore.train.model import Model
from mindspore.context import ParallelMode
from mindspore.nn.metrics import Accuracy
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor
from mindspore.common import set_seed
from Multimodel import TargetNet
from mindspore.train.loss_scale_manager import DynamicLossScaleManager, FixedLossScaleManager
from mindspore.dataset import transforms, vision
from ms_dataset import *

"""get parameters for Momentum optimizer"""
def get_param_groups(network):
    """get parameters"""
    decay_params = []
    no_decay_params = []
    for x in network.trainable_params():
        parameter_name = x.name
        if parameter_name.endswith('.bias'):
            # all bias not using weight decay
            no_decay_params.append(x)
        elif parameter_name.endswith('.gamma'):
            # bn weight bias not using weight decay, be carefully for now x not include BN
            no_decay_params.append(x)
        elif parameter_name.endswith('.beta'):
            # bn weight bias not using weight decay, be carefully for now x not include BN
            no_decay_params.append(x)
        else:
            decay_params.append(x)

    return [{'params': no_decay_params, 'weight_decay': 0.0}, {'params': decay_params}]

def create_dataset(name, batch_size=1, istrain=True):
    batch_size = batch_size
    istrain = istrain
    if name == 'q&a':
        if istrain:
            transform_ms = [
                vision.RandomHorizontalFlip(),
                vision.Resize((224, 224)),
                # vision.ToTensor(),
                # vision.Normalize(mean=(0.485, 0.456, 0.406),
                #                  std=(0.229, 0.224, 0.225)),
                vision.HWC2CHW()
            ]
        else:
            transform_ms = [
                vision.Resize((224, 224)),
                vision.ToTensor(),
                vision.Normalize(mean=(0.485, 0.456, 0.406),
                                 std=(0.229, 0.224, 0.225)),
            ]
    index = list(range(0, 9042))
    data_loader = SingleBenchFolder(index=index)
    my_data = de.GeneratorDataset(data_loader,
                                          column_names=["image", "label1"])
    my_data = my_data.map(input_columns="image", operations=transform_ms)
    my_data = my_data.batch(batch_size, num_parallel_workers=1, drop_remainder=True)
    return my_data

if __name__ == "__main__":
    ds_train = create_dataset(name='q&a', batch_size=4, istrain=True)
    epoch_size = 100
    loss = nn.L1Loss()
    metrics = None
    lr = Tensor(0.001)
    network = TargetNet(16, 224, pretrained=False)
    loss_scale = 1024
    opt = nn.Momentum(params=get_param_groups(network),
                      learning_rate=lr,
                      momentum=0.9,
                      weight_decay=0.0001,
                      loss_scale=loss_scale)
    loss_scale_manager = FixedLossScaleManager(loss_scale, drop_overflow_update=False)

    model = Model(network, loss_fn=loss, optimizer=opt, metrics=metrics, loss_scale_manager=loss_scale_manager)

    ckpt_path = "./ckpt"
    ckpt_save_dir = ckpt_path

    time_cb = TimeMonitor(data_size=ds_train.get_dataset_size())
    config_ck = CheckpointConfig(save_checkpoint_steps=ds_train.get_dataset_size(),
                                 keep_checkpoint_max=10)
    ckpoint_cb = ModelCheckpoint(prefix="checkpoint_alexnet", directory=ckpt_save_dir, config=config_ck)

    print("============== Starting Training ==============")
    model.train(epoch_size, ds_train, callbacks=[time_cb, ckpoint_cb, LossMonitor()],
                dataset_sink_mode=False)

