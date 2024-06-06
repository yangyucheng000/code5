import mindspore as ms

import data
import model
import loss
import option
from trainer.trainer_vidue_worsu_smph import Trainer_UNet
from logger import logger

args = option.args
ms.set_seed(args.seed)
ms.set_context(mode=ms.PYNATIVE_MODE)#ms.GRAPH_MODE)
chkp = logger.Logger(args)

if args.task == 'VideoDeblur':
    print("Selected task: {}".format(args.task))
    model = model.Model(args, chkp)
    loss = loss.Loss(args, chkp) if not args.test_only else None
    loader = data.Data(args)
    t = Trainer_UNet(args, loader, model, loss, chkp)
    while not t.terminate():
        t.train()
        t.test()
else:
    raise NotImplementedError('Task [{:s}] is not found'.format(args.task))

chkp.done()
