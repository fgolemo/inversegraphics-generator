import os
import pickle
import numpy as np
from tqdm import tqdm
from inversegraphics_generator.obj_generator import ObjGenerator

# spotify:track:201ohaBcMqr1k7jzEgBAKh
# https://open.spotify.com/track/201ohaBcMqr1k7jzEgBAKh?si=eq4tI0JEQBiPTL3Rg8Ul_w

# from: https://open.spotify.com/user/spotify/playlist/37i9dQZEVXcHz1uKVd65vg?si=FIkd0LBAS9en1sRXxkH57A

IN_PATH = os.path.expanduser("~/data/ig/aggregated-question.pkl")
OUT_PATH = os.path.expanduser("~/data/ig/iqtest-v1.npz")
PARAMS = [(11, 5), (13, 5), (14, 6), (15, 8)]

MAX_GRID = 8

with open(IN_PATH, "rb") as output_file:
    refs = pickle.load(output_file)
    cola = pickle.load(output_file)
    colb = pickle.load(output_file)
    colc = pickle.load(output_file)
    cold = pickle.load(output_file)
    answ = pickle.load(output_file)

cols = [cola, colb, colc, cold]

og = ObjGenerator(MAX_GRID, 1.0)


def get_grid(in_list):
    obj, idx = in_list
    grid = np.array(obj).reshape((PARAMS[idx][1], PARAMS[idx][1], PARAMS[idx][1]))
    uniform_grid = embed_in_bigger(grid, MAX_GRID)
    return uniform_grid


def embed_in_bigger(grid, bigger_size):
    if grid.shape[0] == bigger_size:
        return grid

    out = np.zeros((bigger_size, bigger_size, bigger_size), dtype=np.uint8)
    out[:grid.shape[0], :grid.shape[0], :grid.shape[0]] = grid
    return out


train = np.zeros((200000, 4, MAX_GRID, MAX_GRID, MAX_GRID), dtype=np.uint8)
test = np.zeros((100000, 4, MAX_GRID, MAX_GRID, MAX_GRID), dtype=np.uint8)
val = np.zeros((len(answ) - 300000, 4, MAX_GRID, MAX_GRID, MAX_GRID), dtype=np.uint8)

for i in tqdm(range(len(answ))):
    if i < 200000:
        train[i,0,:,:,:] = get_grid(refs[i])
        distr = 1
        for j in range(4):
            if answ[i] == j: # ignore the right answer
                continue
            train[i,distr,:,:,:] = get_grid(cols[j][i])
            distr+=1

    if i >= 200000 and i < 300000:
        test[i-200000, 0, :, :, :] = get_grid(refs[i])
        distr = 1
        for j in range(4):
            if answ[i] == j:  # ignore the right answer
                continue
            test[i-200000, distr, :, :, :] = get_grid(cols[j][i])
            distr += 1

    if i >= 300000:
        val[i-300000, 0, :, :, :] = get_grid(refs[i])
        distr = 1
        for j in range(4):
            if answ[i] == j:  # ignore the right answer
                continue
            val[i-300000, distr, :, :, :] = get_grid(cols[j][i])
            distr += 1

np.savez_compressed(OUT_PATH, train=train, test=test, val=val)

