import os
from data import videodata_prior_pretrain


class GOPRO_PRIOR_PRETRAIN(videodata_prior_pretrain.VIDEODATA):
    def __init__(self, args, name='GOPRO', train=True):
        super(GOPRO_PRIOR_PRETRAIN, self).__init__(args, name=name, train=train)

    def _set_filesystem(self, dir_data):
        print("Loading {} => {} DataSet".format("train" if self.train else "test", self.name))
        self.apath = dir_data

