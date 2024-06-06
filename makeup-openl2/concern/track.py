import time

import mindspore as ms


class Track:
    def __init__(self):
        self.log_point = time.time()
        self.enable_track = False

    def track(self, mark):
        if not self.enable_track:
            return
        

        #TODO 监视tensor内存
        print("{} time cost:".format(mark), time.time() - self.log_point)
        self.log_point = time.time()
