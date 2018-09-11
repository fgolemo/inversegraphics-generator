import os
import pickle
import sys

import numpy as np
from anytree import AnyNode, LevelOrderGroupIter
from anytree.exporter import DictExporter
from tqdm import tqdm

from inversegraphics_generator.constants import actions
from inversegraphics_generator.obj_generator import ObjGenerator
from iqtest_objs import get_data_dir

SNEK_LEN = 4
if len(sys.argv) > 1:
    SNEK_LEN = int(sys.argv[1])
GRID_SIZE = 4
OUT_PATH = os.path.join(get_data_dir(),"tree-snek_{}-grid_{}.pkl".format(SNEK_LEN, GRID_SIZE))

upper_bound = np.zeros(1, dtype=np.uint64)
upper_bound[0] = 16
# tmp_factorial = GRID_SIZE ** 3
# for _ in range(SNEK_LEN):
#     upper_bound[0] *= tmp_factorial
#     tmp_factorial -= 1
for _ in range(SNEK_LEN - 1):
    upper_bound[0] *= 6
print("upper bound:", upper_bound[0])

# a = np.eye(5,dtype=np.uint8)
# print (a)
# b = a.tostring()
# print (b)
# c = np.fromstring(b, dtype=np.uint8).reshape((5,5))
# print (c)

og = ObjGenerator(GRID_SIZE, 1.0)

root = AnyNode(id="root")

node_count = 0

# initial layer
for x in range(GRID_SIZE):
    for y in range(GRID_SIZE):
        for z in range(GRID_SIZE):
            node_data = np.zeros((GRID_SIZE, GRID_SIZE, GRID_SIZE), dtype=np.uint8)
            node_data[x, y, z] = 2
            node = AnyNode(parent=root, id=node_data.tostring())
            node_count += 1

depth = 1



with tqdm(total=SNEK_LEN, desc="layer") as pbar_layer, \
        tqdm(total=upper_bound[0], desc="count") as pbar_count:
    pbar_count.update(GRID_SIZE ** 3)  # starting positions
    pbar_layer.update(1)  # starting layer
    while True:
        # get next depth layer of nodes
        nodes = list(LevelOrderGroupIter(root, maxlevel=depth + 1))[-1]

        # this stores the matrices of all nodes that are added one layer below
        new_layer_nodes = []

        for node in nodes:
            node_arr = og.node_to_grid(node)

            # get current position
            pos = np.array(np.where(node_arr == 2)).flatten()

            # list all possible actions from here
            new_poss = [pos + np.array(act) for act in actions]

            for new_pos in new_poss:
                step_valid = og.step_is_valid(node_arr, new_pos)
                if not step_valid:
                    continue

                # create new grid
                tmp_grid = node_arr.copy()

                # set the head to be part of the tail now
                tmp_grid[tmp_grid == 2] = 1

                # add new head
                tmp_grid[new_pos[0], new_pos[1], new_pos[2]] = 2

                identical = False
                symmetric = False

                # get all other nodes of this new layer
                for new_layer_node in new_layer_nodes:

                    if (new_layer_node == tmp_grid).all():
                        identical = True
                        break

                    # flip around 3 axis
                    if (np.flip(new_layer_node, axis=0) == tmp_grid).all() or \
                            (np.flip(new_layer_node, axis=1) == tmp_grid).all() or \
                            (np.flip(new_layer_node, axis=2) == tmp_grid).all():
                        symmetric = True
                        break

                # if node is good, add as child to current
                if not identical and not symmetric:
                    new_layer_nodes.append(tmp_grid)
                    AnyNode(parent=node, id=tmp_grid.tostring())
                    node_count += 1
                    pbar_count.update(1)
        depth += 1
        pbar_layer.update(1)
        if depth == SNEK_LEN:
            break

with open(OUT_PATH, "wb") as output_file:
    pickle.dump(root, output_file)
    print ("wrote output to",output_file)

nodes_last_layer = len(list(LevelOrderGroupIter(root))[-1])

print("nodes in last layer / effective variants:", nodes_last_layer)
