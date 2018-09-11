import pickle
import random

import numpy as np
import anytree
from anytree import LevelOrderGroupIter
import matplotlib.pyplot as plt
from inversegraphics_generator.obj_generator import ObjGenerator

with open("test.pkl", "rb") as infile:
    root = pickle.load(infile)

print (root)
print (len(list(LevelOrderGroupIter(root))[-1]))

og = ObjGenerator(4, 1.0)
last_node = root

while len(last_node.children) > 0:

    next_node = random.sample(last_node.children,1)[0]
    cubes = og.grid_to_cubes(og.node_to_grid(next_node, True))
    og.plot_cube(*cubes)
    plt.show()
    last_node = next_node

