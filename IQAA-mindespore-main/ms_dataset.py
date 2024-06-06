from PIL import Image
import os
import os.path
import scipy.io
import numpy as np
import csv

class SingleBenchFolder():

    def __init__(self, index, patch_num=5):
        import json
        f = open("data/qua&aes.json", encoding="utf-8")

        data_list = json.load(f)
        root = "/home/kaiwei/Dataset/Qua&Aes/dstimgs"

        sample = []
        for i, item in enumerate(index):
            for aug in range(patch_num):
                qua = np.array(float(data_list[item]["qua"])).astype(np.float32)
                aes = np.array(float(data_list[item]["aes"])).astype(np.float32)

                sample.append((os.path.join(root, data_list[item]["image_id"]), qua, aes))

        self.samples = sample

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target1, target2 = self.samples[index]
        sample = pil_loader(path)
        sample = np.array(sample, dtype=np.float32)
        return sample, target1

    def __len__(self):
        length = len(self.samples)
        return length


def getFileName(path, suffix):
    filename = []
    f_list = os.listdir(path)
    for i in f_list:
        if os.path.splitext(i)[1] == suffix:
            filename.append(i)
    return filename


def getTIDFileName(path, suffix):
    filename = []
    f_list = os.listdir(path)
    for i in f_list:
        if suffix.find(os.path.splitext(i)[1]) != -1:
            filename.append(i[1:3])
    return filename


def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')