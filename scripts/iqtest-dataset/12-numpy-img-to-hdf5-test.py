import numpy as np
import matplotlib.pyplot as plt
import h5py
from tqdm import tqdm

IN_PATH1 = "/data/milatmp1/mudumbas/Florian/linear_ambient_test/5000_baseline_data.npy"
IN_PATH2 = "/data/milatmp1/mudumbas/Florian/linear_ambient_test/10000_baseline_data.npy"
OUT_PATH = "/data/lisa/data/iqtest/iqtest-dataset-ambient.h5"


def load(path):
    a = np.load(path)
    a = np.swapaxes(a, 2, 4)  # move color dim to back, but now r/g are swapped
    a = np.swapaxes(a, 2, 3)  # unswap r/g
    a /= 255  # assuming a.max() is around 255 rather than around 1
    a = np.clip(a, 0, 1)
    return a


comb = np.concatenate((load(IN_PATH1), load(IN_PATH2)), axis=0)

# shape: 10000, 4, 128, 128, 3

col_a = comb[:, 0, :, :, :]
col_b = np.roll(comb[:, 1, :, :, :], 2345, axis=0)
col_c = np.roll(comb[:, 2, :, :, :], 666, axis=0)
col_d = comb[:, 3, :, :, :]  # copy of col_a

cols = [col_b,col_c,col_d]

questions = np.zeros((10000,4,128,128,3),dtype=np.float32)
answers = np.zeros((10000),dtype=np.uint8)

order = np.array([0,1,2])

for idx in tqdm(range(10000)):
    questions[idx,0] = col_a[idx]
    np.random.shuffle(order)
    answers[idx] = np.where(order == 2)[0]
    questions[idx, 1] = cols[order[0]][idx]
    questions[idx, 2] = cols[order[1]][idx]
    questions[idx, 3] = cols[order[2]][idx]

with h5py.File(OUT_PATH, "r+") as f:
    test = f["test"]

    labeled_q = test.create_dataset("input",data=questions)
    labeled_a = test.create_dataset("output",data=answers)
