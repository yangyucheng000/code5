import os
import numpy as np
path = "D:\Desktop\AAAI补充材料\LtCMH-master\labels"
files = os.listdir(path)
txts = []
for file in files:
    position = path + '\\'+file
    my_data = np.loadtxt(position)

    print(my_data.sum())




