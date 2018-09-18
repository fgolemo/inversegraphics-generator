import os
import pickle
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from inversegraphics_generator.obj_generator import ObjGenerator
from iqtest_objs import get_data_dir

# spotify:track:201ohaBcMqr1k7jzEgBAKh
# https://open.spotify.com/track/201ohaBcMqr1k7jzEgBAKh?si=eq4tI0JEQBiPTL3Rg8Ul_w

IN_PATH = os.path.expanduser("~/data/ig/aggregated-question.pkl")
PARAMS = [(11, 5), (13, 5), (14, 6), (15, 8)]

with open(IN_PATH, "rb") as output_file:
    refs = pickle.load(output_file)
    cola = pickle.load(output_file)
    colb = pickle.load(output_file)
    colc = pickle.load(output_file)
    answ = pickle.load(output_file)

og0 = ObjGenerator(PARAMS[0][1], 1.0)
og1 = ObjGenerator(PARAMS[1][1], 1.0)
og2 = ObjGenerator(PARAMS[2][1], 1.0)
og3 = ObjGenerator(PARAMS[3][1], 1.0)

ogs = [og0, og1, og2, og3]

def plot_obj_from_list(in_list):
    obj, idx = in_list
    grid = np.array(obj).reshape((PARAMS[idx][1],PARAMS[idx][1],PARAMS[idx][1]))
    v, f = ogs[idx].grid_to_cubes(grid)
    ogs[idx].plot_cube(v,f)
    # plt.show()

def embed_in_bigger(grid, bigger_size):
    out = np.zeros(())

for i in range(10):
    ref = refs[i]
    a = cola[i]
    b = colb[i]
    c = colc[i]
    ans = answ[i]

    plot_obj_from_list(ref)
    plot_obj_from_list(a)
    plot_obj_from_list(b)
    plot_obj_from_list(c)
    print (ans)
    plt.show()


    # print (ref)
    # print ("===")
    # print (a)
    # print ("===")
    # print (b)
    # print ("===")
    # print (c)
    # print ("===")
    # print (ans)
    #
    # quit()



