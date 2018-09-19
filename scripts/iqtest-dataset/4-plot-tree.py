import pickle
import os
import matplotlib.pyplot as plt
from inversegraphics_generator.obj_generator import ObjGenerator
from inversegraphics_generator.iqtest_objs import get_data_dir


SNEK_LEN = 9
GRID_SIZE = 4
OUT_PATH = os.path.join(get_data_dir(),"tree-v2-snek_{}-grid_{}.pkl".format(SNEK_LEN, GRID_SIZE))
print ("loading file:",OUT_PATH)

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

root = pickle.load(open(OUT_PATH, "rb"))

leafs = getLeafs(root)
print ("#leafs:",len(leafs))

for l in leafs:
    print (l)
    grid = og.node_to_grid(l, True)
    v,f = og.grid_to_cubes(grid)
    og.plot_cube(v,f)
    plt.show()
