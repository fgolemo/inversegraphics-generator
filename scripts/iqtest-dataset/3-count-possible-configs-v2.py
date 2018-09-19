import os
import pickle
import sys

import numpy as np
from anytree import AnyNode, LevelOrderGroupIter
from tqdm import tqdm

from inversegraphics_generator.constants import actions
from inversegraphics_generator.obj_generator import ObjGenerator
from inversegraphics_generator.iqtest_objs import get_data_dir

SNEK_LEN = 15
if len(sys.argv) > 1:
    SNEK_LEN = int(sys.argv[1])
GRID_SIZE = 7
OUT_PATH = os.path.join(get_data_dir(),"tree-v2-snek_{}-grid_{}.pkl".format(SNEK_LEN, GRID_SIZE))
BREAK_A = -1
BREAK_B = 6


upper_bound = np.zeros(1, dtype=np.uint64)
upper_bound[0] = 16
# tmp_factorial = GRID_SIZE ** 3
# for _ in range(SNEK_LEN):
#     upper_bound[0] *= tmp_factorial
#     tmp_factorial -= 1
for _ in range(SNEK_LEN):
    upper_bound[0] *= 5
print("upper bound:", upper_bound[0])


og = ObjGenerator(GRID_SIZE, 1.0)

root = AnyNode(id="root")

node_count = 0

# # initial layer - manual init
# node_data = np.zeros((GRID_SIZE, GRID_SIZE, GRID_SIZE), dtype=np.uint8)
# node_data[0, int(math.ceil(GRID_SIZE/2)), 0] = 2
# AnyNode(parent=root, id=node_data.tostring(), remaining=SNEK_LEN-1)
# node_count += 1
#
# node_data = np.zeros((GRID_SIZE, GRID_SIZE, GRID_SIZE), dtype=np.uint8)
# node_data[int(math.ceil(GRID_SIZE/2)), int(math.ceil(GRID_SIZE/2)), int(math.ceil(GRID_SIZE/2))] = 2
# AnyNode(parent=root, id=node_data.tostring(), remaining=SNEK_LEN-1)
# node_count += 1
# initial layer
for x in range(GRID_SIZE):
    for y in range(GRID_SIZE):
        for z in range(GRID_SIZE):
            node_data = np.zeros((GRID_SIZE, GRID_SIZE, GRID_SIZE), dtype=np.uint8)
            node_data[x, y, z] = 2
            node = AnyNode(parent=root, id=og.grid_to_str(node_data), remaining=SNEK_LEN-1)
            node_count += 1



depth = 1

with tqdm(total=upper_bound[0], desc="count") as pbar_count:
    pbar_count.update(GRID_SIZE ** 3)  # starting positions
    while True:
        # get next depth layer of nodes
        nodes = list(LevelOrderGroupIter(root, maxlevel=depth + 1))[-1]

        for node in nodes:
            if node.remaining == 0:
                continue

            node_arr = og.node_to_grid(node)


            for act in actions:
                # get current position
                pos = np.array(np.where(node_arr == 2)).flatten()

                # create new grid
                tmp_grid = node_arr.copy()

                added_len = 0

                while node.remaining-added_len > 0 and \
                        np.random.rand()>((added_len+BREAK_A)/BREAK_B):
                    new_pos = pos + np.array(act)
                    step_valid = og.step_is_valid(node_arr, new_pos)
                    if not step_valid:
                        break

                    # set the head to be part of the tail now
                    tmp_grid[tmp_grid == 2] = 1

                    # add new head
                    tmp_grid[new_pos[0], new_pos[1], new_pos[2]] = 2

                    pos = new_pos.copy()
                    added_len += 1

                if added_len > 0:
                    # if node is good, add as child to current
                    AnyNode(parent=node, id=og.grid_to_str(tmp_grid), remaining=node.remaining-added_len)
                    pbar_count.update(1)

        depth += 1
        if depth == SNEK_LEN:
            break

with open(OUT_PATH, "wb") as output_file:
    pickle.dump(root, output_file)
    print ("wrote output to",output_file)


