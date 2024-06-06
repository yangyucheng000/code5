import os

import time

from torch.backends import cudnn

from dataloder import get_loader
from setup import setup_config, setup_argparser


def train_net(config):
    # enable cudnn
    cudnn.benchmark = True

    data_loader = get_loader(config)
    if config.TRAINING.SOLVER == 'nolocal_nowarp':
        from psgan.solver_nolocald_nowarploss import Solver
    else:
        from psgan.solver import Solver
    
    solver = Solver(config, data_loader=data_loader, device="cuda")
    solver.train()

if __name__ == '__main__':
    args = setup_argparser().parse_args()
    config = setup_config(args)
    print("Call with args:")
    print(config)
    config.defrost()
    config.LOG.LOG_PATH = os.path.join(config.LOG.LOG_PATH, 'window_size_' + str(config.TRAINING.WINDOWS) + "/" + str(int(time.time())))
    # os.makedirs(config.LOG.LOG_PATH, exist_ok=True)
    # os.system('cp {} {}'.format('./psgan/config.py', config.LOG.LOG_PATH))

    config.freeze()
    train_net(config)
