import pickle
import os
from tqdm import tqdm

from inversegraphics_generator.obj_generator import ObjGenerator
from inversegraphics_generator.iqtest_objs import get_data_dir
import numpy as np

SNEK_LEN = 9
GRID_SIZE = 4
OUT_PATH = os.path.join(get_data_dir(), "list-v2-snek_{}-grid_{}".format(SNEK_LEN, GRID_SIZE))

og = ObjGenerator(GRID_SIZE, 1.0)

final = []

with open(OUT_PATH+"-unique.pkl", "rb") as output_file:
    unique = pickle.load(output_file)

for u in tqdm(unique):
    # print (u)

    u_grid = og.str_to_grid(u)

    clean = True

    for i in range(3):
        tmp = og.grid_to_str(og.center_grid(np.rot90(u_grid, k=i + 1, axes=(0, 1))))
        if tmp in final:
            clean = False
            break

    if not clean:
        continue

    for i in range(3):
        tmp = og.grid_to_str(og.center_grid(np.rot90(u_grid, k=i + 1, axes=(0, 2))))
        if tmp in final:
            clean = False
            break

    if not clean:
        continue

    for i in range(3):
        tmp = og.grid_to_str(og.center_grid(np.rot90(u_grid, k=i + 1, axes=(1, 2))))
        if tmp in final:
            clean = False
            break

    if not clean:
        continue

    final.append(u)

with open(OUT_PATH + "-final.pkl", "wb") as output_file:
    pickle.dump(final, output_file)
    print("wrote output to", output_file)
