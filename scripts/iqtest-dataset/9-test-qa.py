import os
import pickle
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from inversegraphics_generator.obj_generator import ObjGenerator
from iqtest_objs import get_data_dir

# spotify:track:201ohaBcMqr1k7jzEgBAKh
# https://open.spotify.com/track/201ohaBcMqr1k7jzEgBAKh?si=eq4tI0JEQBiPTL3Rg8Ul_w

# from: https://open.spotify.com/user/spotify/playlist/37i9dQZEVXcHz1uKVd65vg?si=FIkd0LBAS9en1sRXxkH57A

IN_PATH = os.path.expanduser("~/data/ig/aggregated-question.pkl")
PARAMS = [(11, 5), (13, 5), (14, 6), (15, 8)]

MAX_GRID = 8

with open(IN_PATH, "rb") as output_file:
    refs = pickle.load(output_file)
    cola = pickle.load(output_file)
    colb = pickle.load(output_file)
    colc = pickle.load(output_file)
    cold = pickle.load(output_file)
    answ = pickle.load(output_file)

og = ObjGenerator(MAX_GRID, 1.0)


def plot_obj_from_list(in_list):
    obj, idx = in_list
    grid = np.array(obj).reshape((PARAMS[idx][1], PARAMS[idx][1], PARAMS[idx][1]))
    uniform_grid = embed_in_bigger(grid, MAX_GRID)
    v, f = og.grid_to_cubes(uniform_grid)
    og.plot_cube(v, f)
    # plt.show()


def embed_in_bigger(grid, bigger_size):
    if grid.shape[0] == bigger_size:
        return grid

    out = np.zeros((bigger_size, bigger_size, bigger_size), dtype=np.uint8)
    out[:grid.shape[0], :grid.shape[0], :grid.shape[0]] = grid
    return out


for i in range(10):
    ref = refs[i]
    a = cola[i]
    b = colb[i]
    c = colc[i]
    d = cold[i]
    ans = answ[i]

    plot_obj_from_list(ref)
    plot_obj_from_list(a)
    plot_obj_from_list(b)
    plot_obj_from_list(c)
    plot_obj_from_list(d)
    print(ans)
    plt.show()
