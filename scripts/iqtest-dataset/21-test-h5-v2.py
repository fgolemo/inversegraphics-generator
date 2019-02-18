import random
import time

import numpy as np
import h5py
from tqdm import tqdm
from inversegraphics_generator.iqtest_objs import get_data_dir

OUT_PATH = get_data_dir() + "/3diqtt-v2-x2.h5"

f = h5py.File(OUT_PATH, 'r')

print(list(f.keys()))

q = f["train/labeled/questions"]

print(q.shape)
print(f["train/labeled/answers"].shape)
print(f["train/unlabeled/questions"].shape)

print(f["test/questions"].shape)

print(f["val/questions"].shape)
print(f["val/answers"].shape)

import matplotlib.pyplot as plt
idx = 0

for i in range(3):
    start = time.time()
    idx = random.randrange(0,len(q))
    x, axarr = plt.subplots(1,4, sharex=True, figsize=(20, 10))
    for j in range(4):
        axarr[j].imshow(q[idx,j])
    print(q[idx])
    diff = time.time()-start
    print ("time: ",np.around(diff,4))
    plt.show()
    # idx += 1

## WITH gzip
# file size: 2GB per 10k data
# random access: ~8s per 4 img
# sequential access: ~6s per 4 img

## WITHOUT gzip
# file size: 2.4GB per 10k data
# random access: ~0.2s per 4 img
# sequential access: ~0.2s per 4 img