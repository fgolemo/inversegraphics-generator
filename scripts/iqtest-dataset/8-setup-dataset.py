import os
import pickle
import numpy as np
from tqdm import tqdm

from iqtest_objs import get_data_dir

OUT = os.path.join(get_data_dir(), "aggregated.pkl")
PARAMS = [(11, 5), (13, 5), (14, 6), (15, 8)]

with open(OUT, "rb") as output_file:
    aggr = pickle.load(output_file)

order = np.arange(3)

print(len(aggr))

# og4 = ObjGenerator(GRID_SIZE_4, 1.0)

refs = []
cola = []
colb = []
colc = []
answ = []

for i in tqdm(range(len(aggr))):
    print (aggr[i])

    # take out reference answer
    refs.append(aggr[i][0])

    line_idx = order.copy()

    # randomize order of answers in this line
    np.random.shuffle(line_idx)
    print(line_idx)

    line_tmp = np.array(aggr[i])[line_idx]
    print(line_tmp)
    cola.append(line_tmp[0])
    colb.append(line_tmp[1])
    colc.append(line_tmp[2])

    right_solution_idx = np.where(line_idx == 0)[0][0]
    print (right_solution_idx)

    answ.append(right_solution_idx)

    #TODO: verify that this actually works
    quit()

