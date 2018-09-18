import pickle
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

from inversegraphics_generator.obj_generator import ObjGenerator
from iqtest_objs import get_data_dir
import numpy as np

SNEK_LEN = 9
GRID_SIZE = 4
IN_PATH = os.path.join(get_data_dir(),"tree-v2-snek_{}-grid_{}.pkl".format(SNEK_LEN, GRID_SIZE))
print ("loading file:",IN_PATH)
OUT_PATH = os.path.join(get_data_dir(),"list-v2-snek_{}-grid_{}".format(SNEK_LEN, GRID_SIZE))

og = ObjGenerator(GRID_SIZE, 1.0)

def getLeafs(node):
    if len(node.children) == 0:
        return node
    else:
        out=[getLeafs(child) for child in node.children]
        out_flat = []
        for child in out: # each layer removes some mess
            if type([]) == type(child):
                for child2 in child:
                    out_flat.append(child2)
            else:
                out_flat.append(child)
        return out_flat

root = pickle.load(open(IN_PATH, "rb"))

leafs = getLeafs(root)
print ("#leafs:",len(leafs))

centered = []

for l in tqdm(leafs):
    grid = og.str_to_grid(l.id, True)
    centered.append(og.grid_to_str(og.center_grid(grid)))

def uniqify(seq):
   # Not order preserving
   keys = {}
   for e in tqdm(seq):
       keys[e] = 1
   return keys.keys()

unique = list(uniqify(centered))

print (len(unique))

with open(OUT_PATH+"-unique.pkl", "wb") as output_file:
    pickle.dump(unique, output_file)
    print ("wrote output to",output_file)

# now check for all [90,180,270] deg rotations
# in all directions + centering if it exists.

# final = []
#
# for u in tqdm(unique):
#     u_grid = og.str_to_grid(u)
#
#     clean = True
#
#     for i in range(3):
#         tmp = og.grid_to_str(og.center_grid(np.rot90(u_grid, k=i+1, axes=(0, 1))))
#         if tmp in final:
#             clean = False
#             break
#
#     if not clean:
#         continue
#
#     for i in range(3):
#         tmp = og.grid_to_str(og.center_grid(np.rot90(u_grid, k=i+1, axes=(0, 2))))
#         if tmp in final:
#             clean = False
#             break
#
#     if not clean:
#         continue
#
#     for i in range(3):
#         tmp = og.grid_to_str(og.center_grid(np.rot90(u_grid, k=i+1, axes=(1, 2))))
#         if tmp in final:
#             clean = False
#             break
#
#     if not clean:
#         continue
#
#     final.append(u)
#
# with open(OUT_PATH+"-final.pkl", "wb") as output_file:
#     pickle.dump(final, output_file)
#     print ("wrote output to",output_file)
